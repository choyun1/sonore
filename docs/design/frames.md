# Frames

The design of `sonore.analysis.frames`: what `analyze`, `synthesize` and
`frame_bounds` mean, the frames sonore ships, and the "seeing speech" gallery
section that uses them. The work is split into steps, each agreed here before
it is implemented and delivered as a stack of patches.

| Step | Section | Claims | Decisions | Checker | Status |
|---|---|---|---|---|---|
| 1 | [The interface and its contract](#step-1-the-interface-and-its-contract) | C1–C7 | D1–D4 | `tools/check_frames_step1_claims.py` | Implemented |
| 2 | [New filter shapes, time-varying Gabor, adjoints](#step-2-new-filter-shapes-time-varying-gabor-adjoints) | C8–C14 | D5–D11 | `tools/check_frames_step2_claims.py` | Implemented |
| 3 | [Seeing speech](#step-3-seeing-speech) | C15–C20 | D12–D22 | `tools/check_frames_step3_claims.py` | Implemented |

Claim and decision numbers run on across the steps, so a number names one
claim in this whole file. The plan as first written:

1. **The interface and its contract.** A `Frame` base class. The two
   existing transforms, the cosine filterbanks and the STFT, are retrofitted
   onto it. A dense-matrix oracle is added to the tests.
2. **Two painless families plus unions.** Frequency-domain filterbanks with
   arbitrary shapes (gammatone, Morlet), Gabor frames with time-varying
   windows, and unions of frames that are diagonal in the same domain.
   (Unions were later deferred, D11.)
3. **"Seeing speech."** A gallery section that runs one sentence through
   several frames, plus reassignment for comparison.

The principles behind these choices, in plain words, are in
`docs/design/philosophy.md`.

## Step 1: the interface and its contract

Status: accepted 2026-09-30, with decisions D1–D4 as recommended below. Step 1 is
implemented (`Frame`, `Filterbank`, `GaborFrame`, the cosine-bank and STFT
retrofits), with the dense-matrix oracle tests in tests/test_frames.py.

Step 1 adds no new filter shapes and no new user-visible transforms. Its job
is to fix what `analyze`, `synthesize` and `frame_bounds` mean, precisely
enough that steps 2 and 3 only add instances.

### How the claims are verified

Every mathematical claim is numbered (C1, C2, ...) and tagged with how it is
checked:

- **[proof]**: a finite-dimensional argument, given here, short enough to
  check by hand.
- **[check]**: a number printed by `tools/check_frames_step1_claims.py`. The script
  uses only NumPy and SciPy and builds every operator as an explicit dense
  matrix at N = 64, so it shares no code with the implementation. It runs in
  about a second. The numbers quoted below come from running it with SciPy
  1.17.1.
- **[source]**: a published result, with the citation checked by lookup (see
  References).

### Setting

A sound channel is a vector x in R^N. Channels are always processed
independently, so everything below is per channel.

A frame's *analysis operator* T maps x to coefficients c = Tx. The
coefficients are real (filterbank subbands) or complex (STFT). The coefficient
space carries the real inner product ⟨a, b⟩ = Re Σ_j w_j conj(a_j) b_j, with
weights w_j > 0. For almost every frame, w_j = 1. The one exception is the
half-spectrum STFT; see C3.

The adjoint T* is taken with respect to this inner product. The *frame
operator* is S = T*T, a symmetric positive semidefinite N×N matrix.

### Claims

**C1. Bounds are extreme eigenvalues.** [proof, check]
T is a frame with bounds A ≤ B, meaning A‖x‖² ≤ ‖Tx‖² ≤ B‖x‖² for all x,
with A > 0.

- Proof: ‖Tx‖² = ⟨x, Sx⟩. By the Rayleigh quotient, the tightest constants
  are A = λ_min(S) and B = λ_max(S). These are also the extreme squared
  singular values of T.
- A > 0 exactly when T is injective (full column rank).
- Check: eigenvalues and SVD agree to 4e-16.

**C2. The canonical dual is the pseudo-inverse and gives least squares.**
[proof, check]
Define synthesis as T⁺ = S⁻¹T*.

- Exact reconstruction: T⁺T = S⁻¹S = I.
- Least squares for any coefficients: for arbitrary c, the normal equations
  of min_x ‖Tx − c‖² are T*Tx = T*c. So the unique minimizer is x = T⁺c.
- Projection: P = TT⁺ is self-adjoint and idempotent, so it is the orthogonal
  projection onto range(T). Synthesizing modified coefficients therefore
  equals synthesizing the nearest *consistent* coefficients, meaning the
  nearest coefficients that some signal actually produces.
- Check:
  - S⁻¹T* matches pinv(T) to 3e-15.
  - Exact reconstruction to 1e-15.
  - T⁺c matches lstsq to 3e-15.
  - ‖P² − P‖ + ‖P − Pᵀ‖ = 1e-14.

**C3. Half-spectrum coefficients need weights.** [proof, check]
A real x has conjugate-symmetric DFT frames. A one-sided STFT stores only
bins 0..K/2.

- The full two-sided coefficient energy is recovered by giving every stored
  bin weight 2, except DC (weight 1) and, for even K, Nyquist (weight 1).
- SciPy's one-sided `istft` solves exactly this weighted real least-squares
  problem:
  - It matches the weighted real least-squares solution to 3e-15 on masked
    coefficients.
  - Against the *unweighted* problem it is off by 9%.
- So the frame bounds and the "nearest" in C2 are defined in the weighted
  norm. This applies to any sonore code that measures STFT coefficient
  energy.
- Filterbank subbands are real, because the responses are real and even in
  frequency. They use w = 1.

**C4. Circular filterbanks are diagonalized by the DFT (painless in
frequency).** [proof, check, source: Balazs et al. 2011]
Let T_k = F⁻¹ diag(H_k) F, with H_k(−f) = conj(H_k(f)) so the outputs are
real. Then:

- S = Σ_k T_k*T_k = F⁻¹ diag(s) F, where s(f) = Σ_k |H_k(f)|².
- The eigenvalues of S are s on the DFT grid, so A = min s and B = max s.
- The dual filters are S⁻¹T_k* = F⁻¹ diag(conj(H_k)/s) F, that is, H_k/s.
- For the cosine banks s ≡ 1 by construction. They are tight frames
  (A = B = 1), and the dual filters equal the analysis filters. The current
  `Subbands.synthesize` is therefore already the canonical dual.
- Check, on a deliberately non-tight Gaussian bank:
  - SVD bounds match (min s, max s) to 4e-16.
  - The H/s dual matches pinv on modified coefficients to 4e-15.

**C5. Gabor frames with window length ≤ FFT length are diagonal in time
(painless in time).** [proof, check, source: Daubechies, Grossmann & Meyer
1986]
Let the window have length L ≤ K = n_fft and the hop be a.

- For each frame q, the K DFT samples of a length-≤K segment satisfy
  Parseval: Σ_k |c_{q,k}|² = K Σ_t |x(t) w(t − qa)|².
- Summing over q gives ‖Tx‖² = Σ_t |x(t)|² s(t), where
  s(t) = K Σ_q |w(t − qa)|².
- So S = diag(s), the bounds are min s and max s over the signal, and the
  dual window is w/s.
- `ShortTimeFFT` requires K ≥ L, so every STFT sonore can build is in this
  case. Windows longer than the FFT (time aliasing) are not painless and are
  out of scope.
- Check, one- and two-sided, on a non-tight window (Hann^1.5, hop 5 of 16):
  - The largest off-diagonal entry of S is 5e-17.
  - The eigenvalues match the formula to 2e-15.

**C6. Padding: harmless for Gabor, not for non-tight filterbanks.** [proof,
check]
Let E: R^N → R^M be the zero-padding embedding. The padded analysis is TE,
with frame operator E*SE (a principal submatrix of S).

- Gabor (S diagonal): E*SE is just the diagonal restricted to the signal.
  So "circular dual, then crop" *is* the canonical dual, and padding changes
  nothing. Check: SciPy's `istft` matches the dense canonical least squares
  to 3e-15 on masked coefficients.
- Filterbank (S circulant):
  - E*SE is generally not circulant.
  - Its eigenvalues lie within [min s, max s], because its Rayleigh quotient
    is that of S restricted to padded vectors. In general they lie strictly
    inside.
  - "Circular dual on the padded grid, then crop" is a *left inverse*
    (E*S⁻¹T*TE = E*E = I). So reconstruction of unmodified coefficients is
    still exact.
  - It is the canonical dual *only if s is constant*.
- Check:
  - Non-tight bank: padded bounds are inside the circular ones (margins
    0.019 and 0.018). Crop-of-circular-dual is exact on unmodified
    coefficients (5e-16) but differs from the true canonical dual by 1.3% on
    masked coefficients.
  - Tight bank: the difference is 9e-16.
- This matters from step 2 on, because gammatone and Morlet banks are not
  tight. See decision D1.

**C7. Noise amplification is bounded by 1/√A.** [proof, check]

- σ_min(T) = √A, so ‖T⁺‖ = 1/√A and ‖T⁺c‖ ≤ ‖c‖/√A.
- A small A means that errors in modified coefficients can be amplified on
  synthesis. This is why `frame_bounds` is worth reporting even when
  reconstruction is exact.
- Check: the largest ratio over 200 random c is 0.11 below 1/√A.

**Negative cases.** [check]
`ShortTimeFFT` refuses to build non-frames:

- hop > window length is rejected at construction.
- A periodic Hann window with hop equal to its length is rejected as "not
  invertible", because the window has a zero, so s(t) = 0 there.

sonore's own `frame_bounds` must report A = 0 in these cases without relying
on SciPy's check.

### The contract (to go into the `Frame` docstring)

A `Frame` is a linear transform together with its inverse. It is the
operator, not the coefficients.

- `analyze(sound) -> coefficients`, for example `Subbands` or `STFT`.
- `synthesize(coefficients) -> Sound`. For unmodified coefficients this
  reconstructs exactly, to floating-point precision. For modified
  coefficients (masks, gains, imposed statistics, liftering), it returns the
  least-squares signal: the one whose coefficients are nearest to the given
  ones (C2). With default padding on a non-tight filterbank, "nearest" is
  measured on the padded grid (D1).
- `frame_bounds(n_samples, fs) -> (A, B)` gives the extreme eigenvalues of
  the frame operator for signals of that length and rate. A = B means the
  frame is tight. A = 0 means it is not a frame, and `synthesize` raises. The
  ratio B/A bounds how much coefficient errors can be amplified (C7).
- Coefficients that have lost information, such as magnitudes only, are not
  covered. They need phase retrieval (for example `STFT.griffin_lim`), which
  has no global guarantee.

### Decisions (accepted 2026-09-30)

**D1. Padding with non-tight filterbanks.** C6 shows that "circular dual on
the padded grid, then crop" is exact but not the canonical dual on R^N. There
are three options:

- (a) Define the frame as the circular operator on the padded grid. Report
  its bounds. Document that least squares is on the padded grid, which is
  exact for unmodified coefficients.
- (b) Compute the true canonical dual of the padded operator with CG, using
  the same machinery as texture imposition. This is exact but iterative and
  slower.
- (c) Do (a) by default and offer (b) as an option.

**Decided: (a).** Option (b) will be added in step 2 only if an experiment needs it.
The 1% difference is about where masked energy that lands in the padding
goes, and (a) is honest about it.

**D2. Coefficient norm.** **Decided:** a frame exposes `energy(coefs)`,
which uses the C3 weights, so that bounds, SNRs and tests all use the same
norm.

**D3. Bounds depend on the grid.** For a filterbank, s is sampled on the DFT
grid of the (padded) length. For a Gabor frame, the frame operator is
evaluated over the signal extent. **Decided:** `frame_bounds` takes
`(n_samples, fs)` rather than being a property. For the cosine banks the result is (1, 1)
regardless.

*Implementation note:* `Filterbank.frame_bounds(n_samples, fs, pad="auto")` adds
an optional `pad` matching `analyze`, so the bounds are those of the grid the
coefficients actually live on (D1). `Filterbank.frame_power(n, fs)` exposes s on
the rfft grid of n samples.

**D4. Non-frames.** **Decided:** `analyze` always works. A filterbank with
gaps in coverage is still a fine way to get a cochleagram. `synthesize`
raises when A ≤ 1e-12·B.

A Gabor frame already raises at construction, because SciPy builds the dual
window eagerly. sonore keeps that behavior but re-raises the error with a
clearer message that includes the bounds.

*Implementation note:* `GaborFrame` is defined in seconds, so "construction"
of the SciPy object happens per sampling rate, at the frame's first use at
that rate (`GaborFrame.sft(fs)`, cached). A coverage gap raises there, with
the bounds computed from s(t). A hop longer than the window raises when the
`GaborFrame` itself is constructed. `frame_bounds` never needs SciPy's object,
so it reports A = 0 for a gap frame.

### API sketch

```python
class Frame(ABC):
    def analyze(self, sound: Sound, **kw) -> Coefficients: ...
    def synthesize(self, coefs: Coefficients) -> Sound: ...
    def frame_bounds(self, n_samples: int, fs: float) -> tuple[float, float]: ...
    def energy(self, coefs) -> np.ndarray: ...  # per channel (D2)

class Filterbank(Frame):              # frequency-domain, undecimated (C4)
    tight: bool = False               # True skips the division by s
    def response(self, freqs) -> np.ndarray: ...   # subclasses: (F, n_filters)
    n_filters: int
    cfs: np.ndarray

class CosineFilterbank(Filterbank):   # unchanged behaviour; tight = True
    ...

@dataclass(frozen=True)
class GaborFrame(Frame):              # wraps ShortTimeFFT (C5)
    win_dur: float
    hop_dur: float | None = None      # default win_dur / 4, as now
    window: str = "hann"              # scipy.signal.get_window, periodic
    n_fft: int | None = None          # default: window length
```

As implemented, `window` also accepts a `get_window` tuple or a callable
`n -> array`, which is how a non-standard window such as Hann^1.5 is given
while keeping the frame hashable. `GaborFrame` also exposes `lengths(fs)`,
`window_samples(fs)`, `frame_power(n, fs)` (s(t), summed over every frame that
overlaps the signal; the frames SciPy leaves out overlap only where the window
is zero) and `bin_weights(fs)` (C3). `STFT` gains a keyword `frame=`, and its
private `_data`/`_sft` constructor arguments are gone.

Hann^1.5 is only non-tight for some hops. Its squared window is sin⁶, and a
sum of sin^(2m) over M equal shifts is constant when M > m, so it is tight at
hop = win/4. The checks use hops that don't divide the window evenly.

The `Frame` base, `Filterbank` and `GaborFrame` go in a new module,
`sonore.analysis.frames`. `CosineFilterbank` stays in `sonore.analysis.filterbank` and
subclasses `Filterbank`.

The code is written as pure array functions over the responses and windows,
with no in-place mutation, so that a JAX port is mechanical.

### Changes to existing code

- `CosineFilterbank` gains `n_filters` (= n_bands + 2), `synthesize`,
  `frame_bounds` and `tight = True`. `analyze` is unchanged.
- `Subbands.synthesize()` delegates to `self.filterbank.synthesize(self)`.
  With `tight = True` the arithmetic is the same as today.
- The shape checks in `Subbands` and `Envelopes` use
  `filterbank.n_filters` instead of `n_bands + 2`.
- `Envelopes.modulation_spectrum` needs `spacing` and `unit`, which only
  scale-based banks have. It raises a clear error for other filterbanks.
- `STFT(sound, win_dur, hop_dur)` keeps its signature. It becomes
  `GaborFrame(win_dur, hop_dur).analyze(sound)` and gains a `.frame`
  attribute. `.sft` stays. `to_sound()` delegates to `self.frame.synthesize`.
- `griffin_lim`, `Mask`, `ModulationSpectrum` and the texture code are
  untouched. Texture synthesis only uses circular cosine analysis followed by
  `Subbands.synthesize`, which is bit-for-bit the same path as today.

### Tests (target: under 1 s added)

A helper in tests/helpers.py, `dense_operator(frame, n, fs, pad)`, builds the
analysis matrix by analyzing unit impulses. It includes the padding the fast
path uses, so the oracle tests what the code actually does. A second helper
returns the C3 weights.

For each frame (cosine ERB and octave, a test-only non-tight Gaussian
`Filterbank`, and `GaborFrame` with Hann and a non-tight window), mono and
stereo, at N = 64–128:

- `synthesize(analyze(x)) == x`, with default padding and with `pad=0`.
- `frame_bounds` equals the extreme eigenvalues of the dense, weighted S.
  The cosine banks give (1, 1).
- On masked coefficients, `synthesize` matches the dense canonical least
  squares:
  - Gabor, all paddings.
  - Filterbanks with `pad=0`.
  - For a padded non-tight bank, it matches the documented D1 behaviour
    (circular least squares, then crop), and the test states explicitly that
    this is not the canonical dual.
- Negative cases:
  - A bank with a coverage gap reports A = 0, and `synthesize` raises.
  - Gabor coverage gaps raise at construction.
- Oracle sensitivity: a deliberately broken dual (H instead of H/s, on the
  non-tight bank) must *fail* the least-squares comparison. This keeps the
  oracle honest. It replaces the manual mutation check.
- Regression: the `tight` path and the general H/s path agree to 1e-12 on
  the cosine banks. The existing texture and STFT tests pass unchanged.

### Out of scope for step 1

The following are not part of step 1:

- New filter shapes (gammatone, Morlet).
- Time-varying Gabor windows.
- Unions of frames.
- The CG canonical dual.
- Complex (analytic) subbands. Morlet filters raise the question of whether
  to keep only positive frequencies; that decision belongs to step 2.
- Wrapping the external `nsgt` package.

Step 2's decisions on these are in the step 2 section below.

### References

- Balazs, P., Dörfler, M., Jaillet, F., Holighaus, N. & Velasco, G. (2011).
  Theory, implementation and applications of nonstationary Gabor frames.
  *J. Comput. Appl. Math.* 236(6), 1481–1496. Citation verified by lookup.
  The abstract confirms an explicit canonical dual for the painless case and
  that filterbanks (wavelet, constant-Q) are modeled as nonstationary Gabor
  frames. The specific theorem numbers have not been checked.
- Daubechies, I., Grossmann, A. & Meyer, Y. (1986). Painless nonorthogonal
  expansions. *J. Math. Phys.* 27(5), 1271–1283, doi:10.1063/1.527388.
  Citation verified by lookup.
- Christensen, O. (2003). *An Introduction to Frames and Riesz Bases.*
  Birkhäuser. This is the general reference for C1, C2 and C7. The citation
  is verified as a bibliographic entry, but the chapter and theorem numbers
  have not been checked.
- SciPy `ShortTimeFFT` (≥ 1.12). C3, C5 and C6 were checked against version
  1.17.1.

### Reference implementations

These are for development-time cross-checks only. Their licenses keep them
out of the package: sonore is MIT, LTFAT is GPLv3, and nsgt uses the Artistic
License 2.0.

- [LTFAT](https://ltfat.github.io/) (MATLAB/Octave, GPLv3) is the reference implementation for frames in audio:
  - filterbank, Gabor and nonstationary Gabor frames;
  - canonical duals and frame-bound estimation.

  Its partial Python port, `ltfatpy`, is based on the older LTFAT 2.1.
- [nsgt](https://github.com/grrrr/nsgt) (Python, Artistic License 2.0) is Thomas Grill's port of the
  NUHAG MATLAB code for nonstationary Gabor transforms (Balazs et al. 2011;
  Holighaus et al. 2013). It is the natural cross-check, or optional wrapper,
  for the invertible constant-Q transform.
- SciPy's `ShortTimeFFT` is already a dependency. It is the implementation
  behind `GaborFrame`, and C3, C5 and C6 are checked against it.

## Step 2: new filter shapes, time-varying Gabor, adjoints

Status: accepted, 2026-09-30. D5, D6, D7 and D11 were agreed with Cho in
discussion; D7's transition width, D8, D9 and D10 were accepted as
recommended on 2026-09-30. Implemented in patches 2-4 (see the patch plan).

Step 1 fixed what `analyze`, `synthesize` and
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

### How the claims are verified

As in step 1, claims are numbered (continuing at C8) and tagged [proof],
[check] or [source]. [check] numbers come from
`tools/check_frames_step2_claims.py`, which uses only NumPy and SciPy,
writes every filter response from its formula, and builds operators as dense
matrices on small signals. It shares no code with sonore and runs in about
3 s. The numbers below are from SciPy 1.17.1.

### Claims

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
  minimum `pad="auto"` length, so it costs memory and time, not accuracy.
  (`analyze` rounds that padding up so the FFT length has no prime factor
  above 11: at most a few percent longer, and several times faster than a
  length with a large prime factor.)

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

### Decisions

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
which optimization-based synthesis such as `texture.grad` can use.

**D11. Unions deferred.** (Agreed.) Union synthesis, the adjoints summed and
divided once by the summed s, matters only for editing several
representations jointly and resynthesizing, and step 3 does not do that. D10
makes it nearly free later. STRAIGHT and WORLD combine frames only as power
spectra after phase is discarded, and resynthesize parametrically, so they
don't need it either. A shared time-frequency grid for plotting several
magnitudes side by side belongs with the step 3 plotting work.

### API sketch

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

### Changes to existing code

- `Filterbank.rfft_response`: the Nyquist rule (D8). A no-op for real
  responses.
- `Frame`: a new abstract `adjoint`, implemented for `Filterbank` (one
  method covers all banks), `GaborFrame` and `TVGaborFrame`.
- Edge filters (D7): a shared mechanism for non-tight banks, not a change
  to the cosine banks, which keep their own tight edge filters.
- Exports: `so.GammatoneFilterbank`, `so.MorletFilterbank`,
  `so.TVGaborFrame`.

### Tests (target: under 2 s added)

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

### Patch plan (stacked on 0ba9597, apply in order)

1. This section and `tools/check_frames_step2_claims.py`.
2. `Frame.adjoint` for the existing frames, and the Nyquist rule.
3. `GammatoneFilterbank`, `MorletFilterbank`, and the shared edge filters.
4. `TVGaborFrame` and its coefficient type.

Each patch carries its own tests. README entries come with patches 3 and 4.

### Implementation notes

Details settled while implementing, recorded so the code and this file
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

### Out of scope for step 2

- Union synthesis (D11) and the CG canonical dual (D1(b)).
- Complex-coefficient synthesis (D5).
- Pitch-adaptive schedules, STRAIGHT/WORLD-style envelopes, reassignment,
  and the "seeing speech" gallery (step 3).
- Wrapping `nsgt`.

### References

- Balazs et al. (2011) and Daubechies, Grossmann & Meyer (1986): see
  step 1's references.
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

### Reference implementations

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

## Step 3: seeing speech

Status: implemented, 2026-10-01. Cho accepted the recommendations
for the four open decisions (D12, D13, D17, D18) on 2026-10-01; the other
decisions stand as recommended. Implemented in patches 2–6 (see the
implementation notes below).

Steps 1 and 2 built the frames: the cosine, gammatone and Morlet banks, the STFT, and the
time-varying Gabor frame. Step 3 uses them. It adds one gallery section,
"Seeing speech", that runs one spoken sentence through several analyses and
shows the magnitudes side by side, plus the few small pieces of library code
the section needs:

1. **Pitch-adaptive schedules** for `TVGaborFrame`, including a
   TANDEM-STRAIGHT-style pair.
2. **A reassigned spectrogram,** the one nonlinear analysis, for comparison.
3. **Display-time delay compensation** for causal gammatone plots.
4. **A shared way to draw** every magnitude on the same axes.

### Why this section

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

### How the claims are verified

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

### Claims

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

### Decisions

All recommended. The four that needed Cho's answer are marked
**(accepted 2026-10-01)**; the rest stand as recommended.

**D12. The sentence. (accepted 2026-10-01)** One sentence, about 2–3 s, a male voice
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

**D13. Where F0 comes from. (accepted 2026-10-01)** Pitch-adaptive and TANDEM panels need
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
- **(accepted 2026-10-01)** Which windows the gallery reassigns. Recommended: both the
  5 ms and the 33.3 ms windows, drawn beside their plain spectrograms, so a
  reader sees that reassignment sharpens pulses with a short window and
  harmonics with a long one, but does not escape the choice.
- librosa has a reassigned spectrogram (ISC licence). It is a possible
  dev-time cross-check; status: not yet checked.

**D18. A shared display. (Frequency axis accepted 2026-10-01.)** Every panel in the
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
- **(accepted 2026-10-01)** The frequency axis. Recommended: **linear, 0–5 kHz,** the
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

### Tests (target: under 1 s added)

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

### Patch plan

1. This section and `tools/check_frames_step3_claims.py`.
2. The sentence, its F0 track and `docs/speech/SOURCES.md` (D12, D13).
3. `TVGaborFrame.pitch_adaptive` and `tandem_power` (D15, D16).
4. `reassigned_spectrogram` (D17).
5. Display: `TVSTFT.plot`, the shared axis options, `align` (D18, D19).
6. The gallery section (D20), with README entries.

### Implementation notes

- **Patch 2, the sentence.** `bdl` utterance `arctic_a0131` (2.5 s), chosen
  from three `bdl` files Cho supplied because its Harvest track is the
  cleanest (no octave jumps in the 100 ms overview; F0 about 95–150 Hz,
  falling at the end) and it is the shortest. The files supplied are the
  single-channel versions, without EGG, so the F0 track uses D13's Harvest
  fallback. Details and the licence status are in docs/speech/SOURCES.md.
- **Patch 3, schedules.** `TVGaborFrame.pitch_adaptive(f0_times, f0,
  t_end, periods=3, overlap=4)` and `tandem_power(sound, f0_times, f0,
  periods=2.5, overlap=4, window="blackman")`, which returns a new
  magnitude-only type, `TFPower` (`power`, `t`, `f`, `db`). The schedule
  stops at the last center not after `t_end`, so covering a sound to its
  last sample needs `t_end` = duration plus half the longest window;
  `tandem_power` does that itself. On the sentence (Harvest track,
  3 periods) the frame has 412 windows of 15.6–34.2 ms, A/B 0.78 over the
  whole signal (0.84 in the interior, lowest at 1.69 s where the track
  jumps briefly), and synthesis is exact to 3e-16. That is lower than
  C20's 0.97 because the real track moves faster than C20's glides.
- **Patch 4, reassignment.** `reassigned_spectrogram(sound, frame,
  threshold_db=-60)` returns a `ReassignedSpectrogram` (`t_hat`, `f_hat`,
  `power`, `keep`, all per STFT cell) with `binned(t_edges, f_edges) ->
  TFPower`. The three STFTs are SciPy `ShortTimeFFT`s with the frame's hop
  and FFT length and windows w, τw and w′, each given `dual_win = w` so
  SciPy does not try to build a dual for τw. The window is checked against
  its formula, so a mismatch with SciPy's `get_window` would raise. Tests
  reproduce C18 through the library: off-bin tone within 0.2 Hz, impulse
  exact, chirp within 0.2 Hz of its line, for Hann and Gaussian. On the
  sentence, a 1 ms hop with n_fft 1024 takes 0.5–0.7 s; about half the
  cells survive the −60 dB threshold. librosa's `reassigned_spectrogram`
  is still not cross-checked.
- **Patch 5, display.** `GammatoneFilterbank.envelope_peak_delay`,
  `plot_envelopes(..., align="peak", fscale="linear", fmax=None)`,
  `plot_tf_db`, and `TVSTFT.plot` and `TFPower.plot` on top of it.
  Defaults are unchanged. The README figures regenerate byte-identical to
  main's in the same environment, though neither matches the committed
  PNGs here (matplotlib 3.11.2 in this container), so they were left as
  committed. `align="peak"` uses per-row time coordinates in `pcolormesh`,
  so the drawn shift is exact. The tests check a click: envelope peaks
  spread more than 5 ms uncompensated, and less than 0.2 ms after the
  shift (C19). Each panel's dB range is a plot argument; the gallery
  passes 60 dB (D18), while the existing defaults (80 dB for spectrograms,
  40 dB for cochleagrams) are kept so existing figures don't change.
- **Patch 6, the gallery section.** "Seeing speech" sits after "Speech in
  noise", with the four articles of D20 (keys 27–30). The introduction's
  resynthesis errors are computed at build time on the sound as played; in
  this container they are 2.7e-16 to 1.5e-15 of the peak (largest:
  gammatone). Banks: 54 Morlet wavelets and 60 causal gammatones (2 per
  ERB), 70 Hz to 7 kHz with edge filters, envelopes at 1 kHz; on that
  bank the envelope-peak latency reaches 14 ms, and the article quotes the
  number from the bank. The F0 track is drawn as 10 × F0 so it lies on the
  tenth harmonic instead of along the bottom of a 5 kHz axis. One change to
  D18's numbers: the reassigned points are binned at 5 ms by 20 Hz, not
  1 ms by 10 Hz. Those panels are about 400 by 260 pixels, so a pixel is
  about 6 ms by 19 Hz, and 1 ms bins were drawn by skipping cells, which
  left the reassigned panels speckled; D18's rule (about one pixel) is
  kept. The four articles add about 15 s to the gallery build, within
  D21's estimate. As in patch 5, the existing gallery images regenerate
  differently in this container, so only the new section's files and its
  part of `index.html` are committed.

### Out of scope for step 3

- An F0 estimator in sonore (D13).
- STRAIGHT or WORLD spectral envelopes, aperiodicity, and resynthesis from
  them.
- Union synthesis (step 2's D11), so no synthesis from the TANDEM pair.
- Synchrosqueezing and other nonlinear sharpening besides reassignment.
- Unions and the CG dual, as in step 2.

### References

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
  step 2's references. D13.
- Patterson, R. D. et al. (1992). Complex sounds and auditory images. See
  step 2's references. D19.
- Patterson, R. D., Allerhand, M. H. & Giguère, C. (1995). Time-domain
  modeling of peripheral auditory processing: A modular architecture and a
  software platform. *JASA* 98(4), 1890–1894. doi:10.1121/1.414456.
  Verified from a reference list; the paper itself was not read, so nothing
  about AIM's delay handling is claimed from it. D19.

### Reference implementations

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
