# Frames, step 1: the interface and its contract

Status: accepted 2026-09-30, with decisions D1–D4 as recommended below. Not yet implemented.

The Frame work is split into three steps. This document covers only step 1.

1. **The interface and its contract** (this document). A `Frame` base class.
   The two existing transforms, the cosine filterbanks and the STFT, are
   retrofitted onto it. A dense-matrix oracle is added to the tests.
2. **Two painless families plus unions.** Frequency-domain filterbanks with
   arbitrary shapes (gammatone, Morlet), Gabor frames with time-varying
   windows, and unions of frames that are diagonal in the same domain.
3. **"Seeing speech."** A gallery section that runs one sentence through
   several frames, plus reassignment for comparison.

Step 1 adds no new filter shapes and no new user-visible transforms. Its job
is to fix what `analyze`, `synthesize` and `frame_bounds` mean, precisely
enough that steps 2 and 3 only add instances.

## How the claims are verified

Every mathematical claim is numbered (C1, C2, ...) and tagged with how it is
checked:

- **[proof]**: a finite-dimensional argument, given here, short enough to
  check by hand.
- **[check]**: a number printed by `tools/check_frame_claims.py`. The script
  uses only NumPy and SciPy and builds every operator as an explicit dense
  matrix at N = 64, so it shares no code with the implementation. It runs in
  about a second. The numbers quoted below come from running it with SciPy
  1.17.1.
- **[source]**: a published result, with the citation checked by lookup (see
  References).

## Setting

A sound channel is a vector x in R^N. Channels are always processed
independently, so everything below is per channel.

A frame's *analysis operator* T maps x to coefficients c = Tx. The
coefficients are real (filterbank subbands) or complex (STFT). The coefficient
space carries the real inner product ⟨a, b⟩ = Re Σ_j w_j conj(a_j) b_j, with
weights w_j > 0. For almost every frame, w_j = 1. The one exception is the
half-spectrum STFT; see C3.

The adjoint T* is taken with respect to this inner product. The *frame
operator* is S = T*T, a symmetric positive semidefinite N×N matrix.

## Claims

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

## The contract (to go into the `Frame` docstring)

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

## Decisions (accepted 2026-09-30)

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

**D4. Non-frames.** **Decided:** `analyze` always works. A filterbank with
gaps in coverage is still a fine way to get a cochleagram. `synthesize`
raises when A ≤ 1e-12·B.

A Gabor frame already raises at construction, because SciPy builds the dual
window eagerly. sonore keeps that behavior but re-raises the error with a
clearer message that includes the bounds.

## API sketch

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

The `Frame` base, `Filterbank` and `GaborFrame` go in a new module,
`sonore.frames`. `CosineFilterbank` stays in `sonore.filterbank` and
subclasses `Filterbank`.

The code is written as pure array functions over the responses and windows,
with no in-place mutation, so that a JAX port is mechanical.

## Changes to existing code

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

## Tests (target: under 1 s added)

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

## Out of scope for step 1

The following are not part of step 1:

- New filter shapes (gammatone, Morlet).
- Time-varying Gabor windows.
- Unions of frames.
- The CG canonical dual.
- Complex (analytic) subbands. Morlet filters raise the question of whether
  to keep only positive frequencies; that decision belongs to step 2.
- Wrapping the external `nsgt` package.

## References

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

## Reference implementations

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
