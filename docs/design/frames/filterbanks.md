# Filterbanks: one general bank

The redesign of `sonore.frames.filterbank`, together with two changes the
code audit moved here: subbands go back to a sound with `to_sound`, and
masks become views.

Status: proposed 2026-10-03, awaiting Cho's decisions (D1–D12). Nothing
in `src/` changes until they are agreed. The claims are checked by
`tools/check_filterbank_claims.py` (C1–C8), which uses only NumPy and
SciPy, with every scale and filter written out from its formula, so it
shares no code with the implementation it will later test.

## Why

The file was discussed in the audit (issue #48, sitting 8) out of order.
Cho's position, 2026-10-03:

- The generality the mathematics allows should be kept. There should be
  one general filterbank. Parameters are set where the bank is used, or by
  a function that returns the particular bank wanted, as
  `sources/waveforms.py` does: `pure_tone` and `harmonic_complex` return
  a `Sound`.
- Tightness should be computed, not declared.
- Nobody depends on the API yet, so there is no version bump and no
  aliases. Other redesigns may follow and collect on main.

Today the file has seven classes for what is one idea:

| Class | What varies |
|---|---|
| `Filterbank` | the abstract frame: `response`, `cfs`, `n_filters`, a declared `tight` flag |
| `CosineFilterbank` | half-cycle cosines on some scale, plus a flat lowpass and highpass |
| `ERBFilterbank`, `OctaveFilterbank` | the scale only (two static methods each) |
| `BandpassFilterbank` | any bandpass shape, plus raised-cosine edge filters |
| `GammatoneFilterbank`, `MorletFilterbank` | the shape and its scale |

Each class is frozen, so a bank is fixed once built, which is right, but
the choices that should be arguments (the scale, the shape, the centers,
whether there are edge filters) are classes instead. The same two numbers
also mean different things: for the cosine banks `f_lo` and `f_hi` are
where the flat edge filters begin, with the bandpass filters inside, while
for the bandpass banks they are the lowest and highest bandpass centers,
with the edges outside. Tightness is a class attribute the author promises
(`tight = True`), not a property the code checks.

## What the mathematics allows

An undecimated filterbank applied on the DFT grid is a frame whose frame
operator is diagonal in frequency, `s(f) = sum_k |H_k(f)|^2`. Any set of
responses with `0 < A <= s(f) <= B` is a frame, its canonical dual filters
are `H_k / s`, and it is tight exactly when `s` is constant (Balazs et
al., 2011, call such frames "painless"). Nothing in that statement depends
on the scale, the shape or the spacing. So one class can hold all of them,
and tightness is a fact about `s` that the bank can measure.

## Claims

Each claim gives the number `tools/check_filterbank_claims.py` printed on
2026-10-03 (16 kHz unless stated).

- **C1. Cosine banks are tight on any increasing scale, up to rounding.**
  With 30 bands between 50 Hz (125 Hz on the octave scale) and 7.6 kHz
  on a 1 s grid, the largest `|s - 1|` is 3.2e-15 (ERB), 1.5e-14
  (octave), 6.2e-15 (mel) and 3.6e-15 (linear). `s` equals 1 exactly in
  only 15–62% of the bins. A measured test of tightness therefore needs a
  tolerance; an exact test would call today's banks not tight.
- **C2. Arbitrary centers stay tight.** With 32 random centers on the ERB
  scale and the gap formula (between centers `k` and `k + 1`, with `u` the
  fractional position, filter `k` is `cos(pi u / 2)` and filter `k + 1` is
  `sin(pi u / 2)`), the largest `|s - 1|` is 4.4e-16.
- **C3. The gap formula is not bit-identical to today's on uniform
  centers.** 4.0% of the responses differ, by at most 3.1e-15. Texture
  output must stay bit-identical, so uniform centers keep today's formula
  (distance in units of the one spacing) and the gap formula is used only
  for centers that are not uniform.
- **C4. Centers must be stored on the scale, not in Hz.** Converting
  uniform ERB centers to Hz and back changes 12.5% of them in the last
  bits, and then 0.65% of the responses. A bank therefore keeps its
  centers in scale units and reports Hz only as `cfs`.
- **C5. Wider cosines.** With each cosine `w` spacings wide (today `w =
  1`), `s` between the inner centers is constant, and equal to `w`, when
  `2w` is a whole number of at least 2: `w` = 1.5 and 2 give `s` = 1.5 and
  2 exactly. Other widths ripple: `w` = 0.75 gives 0.5 to 1, 1.25 gives
  1.19 to 1.31, 2.5 gives 2.41 to 2.5. So "tight" should mean "`s` is
  constant", with the bound `A = B = s` reported, not "`s` is 1".
- **C6. The gammatone envelope's peak delay.** For a causal 4th-order
  gammatone the formula `(order - 1) / (2 pi b)` (13.2 ms at 100 Hz,
  0.70 ms at 6 kHz) and the peak measured from the impulse response
  differ by at most 0.076 ms at 16 kHz and 0.072 ms at 44.1 kHz, that is,
  by at most about one sample at 16 kHz.
- **C7. The octave modulation bank is a cosine bank.** The texture
  model's `OctaveModulationFilterbank` (7 bands, one octave apart, up to
  100 Hz) is bit-identical to a cosine bank on log2 frequency with spacing
  one octave and no edge filters: largest difference 0, no response
  differs in any bit.
- **C8. Measuring `s` is cheap.** For 5 s at 44.1 kHz and 32 filters,
  computing `max |s - 1|` takes 5% of the time of one synthesis (4% in an
  earlier run; timings vary by machine and run).

## The proposed design

### One class, built by functions

```python
bank = so.cosine_filterbank(30, 50, 8000)                    # today's ERBFilterbank(30, 50, 8000)
bank = so.cosine_filterbank(30, 125, 8000, scale="octave")   # today's OctaveFilterbank
bank = so.cosine_filterbank(f_lo=125, f_hi=8000, spacing=1/6, scale="octave")  # bands per octave
bank = so.cosine_filterbank(centers=[200, 450, 1000, 2400], scale="erb")       # any centers
bank = so.gammatone_filterbank(30, 50, 8000, phase="causal")
bank = so.morlet_filterbank(30, 50, 8000, cycles=6)
bank = so.Filterbank(scale, centers, shape)                  # the general form
```

`Filterbank` is one frozen class holding three things:

- a **scale**: the frequency axis the centers are spaced on (ERB number,
  octaves, mel, linear), with its two conversions taken from `core/utils`
  (one copy of each formula);
- the **centers**, in scale units (C4), with `cfs` giving them in Hz;
- a **shape**: the response of one filter as a function of its center
  (cosine of a given width, gammatone of a given order, bandwidth factor
  and phase, Morlet of a given number of cycles), plus whether the bank
  has edge filters.

The factories choose sensible arguments and return a `Filterbank`, as
`pure_tone` returns a `Sound`. `so.subbands(sound)` stays as the one-line
convenience. There is no subclass per scale or per shape.

### Tightness is measured

`frame_bounds(n, fs)` already computes `s` for the non-tight banks; it
now does so for every bank. Synthesis measures `s` on the grid the
coefficients live on. When `max |s - c| <= 1e-12 c` for a constant `c`,
it takes the tight path: re-filter, divide by `c` only when `c != 1`. C1
shows today's cosine banks pass with three orders of magnitude to spare,
and since the tight path is the same arithmetic as today's, texture
output stays bit-identical. The cost is about 5% of a synthesis (C8);
caching `s` per grid length and rate, as the ringing time is cached
already, removes it from repeated syntheses such as texture iterations.
`bank.is_tight(n, fs)` reports the result, so the claim "cosine banks are
tight" becomes a test instead of an attribute.

### Edge filters and what `f_lo`, `f_hi` mean

Every factory reads `f_lo` and `f_hi` the same way: the frequencies where
the edge filters' flat part ends. The `n_bands` bandpass centers lie
strictly inside, at the interior points of `n_bands + 2` equally spaced
points from `f_lo` to `f_hi` on the scale. This is today's cosine
convention. For gammatone and Morlet banks it is also exactly where
today's edge filters already put their corners (one spacing outside the
outermost centers), so only the arguments move, not the geometry.

`edges=True` is the default. Cosine banks keep their flat lowpass and
highpass, which is what makes them tight. Other shapes keep today's
raised-cosine edges at the level `sqrt(s_floor)`. `edges=False` gives the
bare bank (a cochleagram with no edge bands), which may not be a frame;
`synthesize` refuses then, as today.

### Other properties

- `spacing` and `scale.unit` exist on every bank with uniform centers, so
  `Envelopes.modulation_spectrum` checks for uniform centers instead of
  for two attributes with a `getattr` fallback.
- Ripples are defined in octaves. `ripples.render` checks
  `bank.scale.name == "octave"` instead of `isinstance(bank,
  OctaveFilterbank)`.
- `envelope_peak_delay` (used by `plot_envelopes(align="peak")`) is
  measured from each filter's impulse response, with parabolic
  interpolation between samples, and cached like the ringing time. This is
  one copy that works for every shape; the gammatone formula stays as a
  test oracle (C6 shows the two agree to within about a sample).

### Subbands go back with `to_sound`

`Subbands.synthesize()` is renamed `Subbands.to_sound()`, matching
`STFT.to_sound()` and the views, so every coefficient object in sonore
goes back to a sound the same way. The frame's own `bank.synthesize(coefs)`
stays (it is the `Frame` contract). A search finds 14 lines that call it in `src/`,
`docs/` and `tests/`; the vocoder idiom becomes
`(speech.envelopes() * carrier.tfs()).to_sound()`.

### Masks become views

A mask is a set of gains on a frame's coefficients. It holds no sound:
many target–masker pairs give the same mask, and none can be read back
from it. That is the definition of a view in `docs/design/philosophy.md`
("Views may discard information"), so `Mask` moves to `views/mask.py` as
a `View`:

- `discards`: "A mask holds gains between 0 and 1 for a frame's
  coefficients, not the coefficients themselves: it has no level or phase
  of any sound."
- `back_to_sound`: "Multiply a frame's coefficients by it and go back from
  those: `(mask * stft).to_sound()`."
- `mask.to_sound()` and `mask.synthesize()` refuse with that message.
- A mask applies to any coefficients on its grid: `STFT`, `TVSTFT` and
  now `Subbands`. `ideal_binary_mask(target, masker)` and
  `ideal_ratio_mask` accept any matching pair. For complex coefficients the
  power is `|X|^2`. For subbands, which are real and oscillate, it is the
  Hilbert envelope squared, so the mask does not flicker at twice each
  band's frequency.
- `Mask.__mul__` is removed. The audit (sitting 7) showed it is dead:
  `mask * stft` already reaches `STFT.__rmul__`.

The docstrings cite where the masks come from:

- The ideal binary mask as the goal of computational auditory scene
  analysis: Wang (2005).
- The local SNR criterion `lc_db`, and its use with listeners: Brungart,
  Chang, Simpson & Wang (2006).
- Ratio masks: Srinivasan, Roman & Wang (2006). The form
  `(S^2 / (S^2 + N^2))^beta` with `beta = 0.5` as the default: Wang,
  Narayanan & Wang (2014), who found 0.5 best and note it resembles the
  square-root Wiener filter.

### Gabor frames stay as they are

`GaborFrame` and `TVGaborFrame` keep their classes (Cho agreed 2026-10-03,
sitting 7). An STFT is a filterbank too: decimated, with identical filters
spaced linearly in frequency. `TVGaborFrame` adapts the window across
time, the general filterbank adapts the filters across frequency, and the
nonstationary Gabor transform (Balazs et al., 2011) does either with one
set of formulas. That would give complex ERB and constant-Q spectrograms
with far fewer coefficients than an undecimated bank, and one copy of the
frame algebra. It is the direction, not this redesign: what this design
does now is avoid anything that would block it later, chiefly by keeping
`hop` out of `Filterbank` rather than fixing it at one sample for every
band in the API.

### Modulation banks, later

C7 shows the texture model's octave modulation bank is a cosine bank on
log2 frequency without edges, bit for bit. It could be built by the same
factory once that exists. The modulation banks filter envelopes for the
texture statistics and are not frames, so this redesign only records the
equivalence; moving them is a separate step, with the texture
bit-identity check.

## Citations

The cosine filters cite McDermott & Simoncelli (2011) alone. They used
this construction on the ERB scale for sound texture, which is why sonore
has it, but the idea is older. Filters whose squared responses sum to one
are called power complementary, and raised-cosine filters of this kind
are the radial filters of the steerable pyramid (Simoncelli & Freeman,
1995; Portilla & Simoncelli, 2000). The docstring will name the
construction and credit both. Where it first appeared is not settled here.

| Source | Status on 2026-10-03 |
|---|---|
| Wang (2005), "On ideal binary mask as the computational goal of auditory scene analysis", in Divenyi (Ed.), *Speech Separation by Humans and Machines*, pp. 181–197, Springer. doi:10.1007/0-387-22794-6_12 | Bibliographic details checked |
| Brungart, Chang, Simpson & Wang (2006), JASA 120(6), 4007–4018. doi:10.1121/1.2363929 | Bibliographic details checked on Crossref; the LC wording not read in the paper |
| Srinivasan, Roman & Wang (2006), "Binary and ratio time-frequency masks for robust speech recognition", Speech Communication 48(11), 1486–1501. doi:10.1016/j.specom.2006.09.003 | Checked |
| Wang, Narayanan & Wang (2014), "On training targets for supervised speech separation", IEEE/ACM TASLP 22(12), 1849–1858 | Checked, including `beta = 0.5` |
| Balazs, Dörfler, Jaillet, Holighaus & Velasco (2011), "Theory, implementation and applications of nonstationary Gabor frames", J. Comput. Appl. Math. | Title and authors found (HAL record); volume and pages from memory (236(6), 1481–1496), to check |
| McDermott & Simoncelli (2011), "Sound texture perception via statistics of the auditory periphery: evidence from sound synthesis", Neuron | Found; volume and pages (71(5), 926–940), and their description of the filters, from memory, to check |
| Portilla & Simoncelli (2000), "A parametric texture model based on joint statistics of complex wavelet coefficients", IJCV 40(1), 49–71 | Bibliographic details found; the raised-cosine radial filters from memory, to check |
| Simoncelli & Freeman (1995), "The steerable pyramid: a flexible architecture for multi-scale derivative computation", ICIP | From memory, to check |

The ones marked "to check" are checked before the docstrings are written.
The page reads that would check them need approval in this project.

## Decisions

Each has a recommendation.

- **D1. One `Filterbank` class** holding a scale, centers on the scale and
  a shape; `CosineFilterbank`, `ERBFilterbank`, `OctaveFilterbank`,
  `BandpassFilterbank`, `GammatoneFilterbank` and `MorletFilterbank` are
  removed, with no aliases. Recommended.
- **D2. Factories** `so.cosine_filterbank`, `so.gammatone_filterbank`,
  `so.morlet_filterbank`, each with a `scale=` argument (defaults: ERB,
  ERB, octave). Recommended.
- **D3. Scales** as small objects named by a string (`"erb"`, `"octave"`,
  `"mel"`, `"linear"`), built from the `core/utils` conversions.
  Recommended. The alternative, passing two functions, is more general
  but loses the name and unit that ripples and modulation spectra need.
- **D4. Centers stored on the scale** (C4); uniform centers use today's
  formula and others the gap formula (C2, C3). Recommended; needed for
  bit-identity.
- **D5. Tightness measured** with a relative tolerance of 1e-12 on a
  constant `s`, cached per grid; the `tight` attribute goes. Recommended.
- **D6. Cosine width** as an argument, `width=1` by default, with the
  bank reporting `s = width` when `2 * width` is a whole number (C5).
  Optional; recommended only because it costs one argument and makes the
  redundant cosine banks used in later texture work a one-liner. Say no
  and it stays out.
- **D7. `f_lo` and `f_hi` mean the same for every shape**: where the edge
  filters' flat part ends, bandpass centers inside. Recommended.
- **D8. `envelope_peak_delay` measured** for every bank, the gammatone
  formula kept as a test. Recommended.
- **D9. `Subbands.synthesize` renamed `to_sound`**, no alias.
  Recommended (Cho asked for it, sitting 7).
- **D10. `Mask` becomes a `View`** in `views/mask.py`, applies to `STFT`,
  `TVSTFT` and `Subbands`, uses Hilbert-envelope power for subbands, and
  loses the dead `__mul__`. Recommended (Cho approved the idea, sitting 7).
- **D11. Citations** as in the table above, after the open ones are
  checked. Recommended.
- **D12. Out of scope now**: the nonstationary Gabor transform, a
  per-band hop, and moving the modulation banks. Recommended.

## Plan

After the decisions, as a stack of PRs:

1. `Filterbank`, scales, factories and measured tightness, with every
   call site moved (`texture/stats.py`, `views/envelopes.py`,
   `views/modulation.py`, `sources/ripples.py`, `spatial/reverb.py`,
   `spatial/binaural.py`, the gallery scripts) and the filterbank tests
   ported, keeping the dense-matrix oracle tests. Texture synthesis is
   hashed against main before and after and must match bit for bit, as
   must the cochleagrams, ripples and modulation spectra.
2. `Subbands.to_sound` and masks as views, with the mask citations.
3. Documentation: the CHANGELOG, README source links, and the gallery
   pages' text patched in place (no regenerated images).

The audit's sitting 8 then reads the new file.
