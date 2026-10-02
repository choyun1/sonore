# Changing a voice: pitch and formants

The design of two changes to a recorded voice, made with the WORLD pieces
already in sonore (`so.f0_track`, `so.cheaptrick`, `so.d4c`,
`so.harmonic_aperiodicity`, `so.world_synthesize`; see `world.md`):

- a **pitch change**, which multiplies the F0 track and leaves the
  spectral envelope alone, so the formants stay where they were;
- a **formant shift**, which moves the spectral envelope along frequency
  and leaves the F0 track alone, so the pitch stays where it was.

Together they are the two numbers of the classic "change gender" manipulation
(Praat's command of that name takes a formant shift ratio and a new pitch
median, among others). The question the design has to settle beyond the
two operations is what happens to the aperiodicity when the envelope moves.

Status: proposal, 2026-10-02. No library code until Cho answers the
decisions below. Every number is printed by `tools/check_voice_change_claims.py`.
The checker measures what the library's own analyses see; whether the
changed voices sound right is for Cho to judge by listening, and nothing
here claims it.

## What exists already

- `world_synthesize` takes any F0 track, and its docstring already says a
  changed F0 is a pitch shift with the envelope and aperiodicity kept. So a
  pitch change needs no new code: `(track.t, track.f0 * 1.5)`. What is
  missing is a statement of what it does to the measured pitch and
  formants (C2), and a named form if Cho wants one (D2).
- `so.pitch_shift` (phase vocoder) changes pitch by stretching and
  resampling, which moves the formants with the pitch (C2 measures it).
- There is no formant shift anywhere in sonore, and no gallery page or
  section changes a voice with WORLD. `world.md` D9 listed "pitch change,
  formant shift, breathiness" as possible demos; the aperiodicity page did
  not include them.
- `SpectralEnvelope` and `Aperiodicity` are callable, `env(t, f)`, reading
  the view at any times and frequencies (linear in time, linear in dB
  between bins). A linear formant shift is that call at `f / ratio`, so the
  prototype in the checker is two lines.

## How the claims are verified

`tools/check_voice_change_claims.py` imports sonore (the WORLD ports are
verified against WORLD in `world.md`) and prototypes the operations
locally:

- pitch change: voiced F0 values times a ratio, 0 stays 0;
- formant shift: `new(f) = old(f / ratio)`, read with the view's own
  interpolation; above fs/2 the value at fs/2 is held;
- aperiodicity: either kept, or warped the same way as the envelope.

Sounds: the bdl sentence `docs/speech/bdl_arctic_a0131.flac` with
`so.f0_track` (hop 5 ms) and D4C; the same sentence by slt; and the
synthetic vowel of `tools/check_female_voices.py` (men's "hod" from
Hillenbrand et al. 1995, F0 120 Hz, envelope and noise share known exactly,
the noise share a straight line from −30 dB at 0 Hz to −5 dB at 8 kHz).

Formants are measured two ways. The **fitted warp** is the ratio *a* that
best explains the changed envelope as the original read at `f / a` plus a
level (level-free RMS over 100–5000 Hz, grid of 0.1% steps from 0.45 to
2.25), both envelopes measured by CheapTrick and averaged in dB over voiced
time windows. The **peaks** are the highest local maxima of that average
within 15% of each expected formant. The fitted warp uses the whole
envelope, so it is the steadier of the two.

The checker runs in about 40 s (NumPy 2.4.6, SciPy 1.17.1).

## Claims

**C1. Identity settings reproduce `world_synthesize` only if ratio 1 skips
the interpolation.** Reading the envelope through its interpolation at the
exact bin frequencies changes it by up to 2.2e−16 (relative), because the
interpolation runs through `exp(log(x))`; the output then differs from
`world_synthesize` by up to 1.4e−16 (peak 0.82). When ratio 1 returns the
envelope itself, the output is identical, sample for sample. F0 times 1.0
is the same array. [check]

**C2. A pitch change moves the measured F0 by the ratio and leaves the
formants in place; the phase vocoder's moves both.** On the bdl sentence:

| ratio | median F0 out/in | within 5% | voiced kept | fitted warp | `so.pitch_shift` warp |
|---|---|---|---|---|---|
| 0.500 | 0.501 | 100% | 17% | 1.005 | 0.501 |
| 0.749 (−5 st) | 0.748 | 100% | 85% | 1.003 | 0.749 |
| 1.498 (+7 st) | 1.496 | 97% | 96% | 1.003 | 1.500 |
| 2.000 | 1.995 | 98% | 98% | 1.002 | 1.999 |

"Voiced kept" is the share of the input's voiced time windows that
`so.f0_track` still calls voiced in the output. At ratio 0.5 most of bdl's
F0 (median 119 Hz) falls below the tracker's default floor of 60 Hz, so
the low share is the tracker's range, not the change. The output envelope
is measured with the imposed track. [check]

**C3. A formant shift moves a synthetic vowel's formants by the ratio.**
CheapTrick sees the input's F1–F3 at 719, 1312 and 2516 Hz (true 768, 1333,
2522).

| ratio | F1, F2, F3 out/in (peaks) | fitted warp (residual) |
|---|---|---|
| 0.80 | 0.826, 0.821, 0.814 | 0.802 (0.73 dB) |
| 0.90 | 1.000, 0.917, 0.907 | 0.900 (0.76 dB) |
| 1.10 | 1.174, 1.095, 1.099 | 1.100 (0.46 dB) |
| 1.20 | 1.174, 1.190, 1.193 | 1.199 (0.41 dB) |
| 1.25 | 1.326, 1.274, 1.242 | 1.250 (0.44 dB) |

The F1 peak ratios jump in steps because CheapTrick's peak near F1 sits on
a harmonic of 120 Hz: 719 Hz is the 6th harmonic, it stays there at 0.9,
and moves to the 7th (840 Hz, ratio 1.17) at 1.1 and 1.2. The fitted warp,
which uses the whole envelope, matches the ratio to within 0.002. [check]

**C4. On the bdl sentence the fitted warp matches the ratio, and the level
moves by up to 1.5 dB.**

| ratio | fitted warp (residual) | output RMS vs identity | envelope power vs identity (median) |
|---|---|---|---|
| 0.80 | 0.801 (0.46 dB) | −1.48 dB | −0.97 dB |
| 0.90 | 0.900 (0.39 dB) | −1.31 dB | −0.47 dB |
| 1.10 | 1.102 (0.38 dB) | −0.25 dB | +0.40 dB |
| 1.20 | 1.203 (0.29 dB) | +0.35 dB | +0.77 dB |

The level changes because the warp stretches or squeezes the envelope's
area: lowering the formants compresses the strong low-frequency part into
fewer bins. At ratio 0.8 the top bins read up to 10 kHz, above fs/2 =
8 kHz, so the top 20% of the band holds the value at fs/2. [check]

**C5. Whether the aperiodicity should move with the envelope is not
settled by the data.**

- *Where the noise is a property of the source*, as in the synthetic vowel,
  keeping the aperiodicity is right: after a shift of 1.2, the
  harmonic-residual aperiodicity measured on the output is within 0.7 dB
  (median, 200–6000 Hz) of the source's noise share when it is kept, and
  1.4 dB when it is warped.
- *On bdl, the harmonic-residual measure finds more noise in the envelope's
  valleys*: across 100–6000 Hz in voiced time windows, the correlation of
  envelope dB and aperiodicity dB, with a straight line in frequency taken
  out of each so a shared tilt does not count, has median −0.33. On the
  synthetic vowel the same measure gives −0.05, so the relation is in the
  recording, not made by the estimator. If that noise is tied to the
  valleys (for example noise not shaped by the same resonances as the
  harmonics), warping keeps it in the valleys; keeping it puts noisier
  bands where the shifted formants now are.
- *D4C shows no such relation on bdl* (median +0.02). At 16 kHz D4C
  measures a single band (`world.md`), and on the synthetic vowel it gives
  +0.62 where the truth is 0, so its shape follows its own band layout
  rather than the voice.
- *How much it matters:* at ratio 1.2 the difference between the warped-
  and kept-aperiodicity outputs is −28 dB relative to the output with D4C,
  and −15 dB with the harmonic residual. [check]

**C6. One pitch ratio and one warp do not turn bdl into slt, but they
close about a quarter of the distance.** On the same sentence, median voiced F0 is
119 Hz for bdl and 183 Hz for slt (ratio 1.53). The warp that best fits
bdl's mean voiced envelope to slt's is 1.65 (residual 4.41 dB, against
6.31 dB unwarped); much of that is a difference in spectral slope, because
allowing a straight line in dB against log frequency as well gives 1.22
(residual 2.99 dB). Hillenbrand et al.'s women-to-men ratios of F1–F3 for
heed, hod and who'd run from 1.11 to 1.28, geometric mean 1.174. Applied to
bdl, the mean voiced envelope's distance to slt's (level removed, 100–5000
Hz) goes from 6.24 dB with F0 alone to 4.54 dB with warp 1.65, 4.97 dB
with 1.22 and 4.86 dB with 1.174. These are distances between average
envelopes, not a statement about how the result sounds. [check]

## Proposed design

Two operations on the views, which are what WORLD's analysis returns, so
the synthesis stays the exact port and every change is visible in the
calling code:

```python
track = so.f0_track(snd)
env = so.cheaptrick(snd, track)
ap = so.d4c(snd, track)

higher = so.world_synthesize(track.scale(1.5), env, ap)          # pitch only
shorter_tract = so.world_synthesize(track, env.warp(1.2), ap)    # formants only
both = so.world_synthesize(track.scale(1.53), env.warp(1.17), ap)
```

- `F0Track.scale(ratio, *, range=1.0)` returns a new `F0Track` with voiced
  values multiplied by `ratio` and, if `range` is not 1, spread around
  their median on a log scale:
  `median * ratio * (f0 / median) ** range`. (Praat's "Change gender"
  has a "pitch range factor"; this definition is sonore's, not read from
  Praat.) Unvoiced windows stay 0. A
  `(times, f0)` pair is changed with NumPy, as now.
- `SpectralEnvelope.warp(ratio)` returns a new envelope with
  `new(f) = old(f / ratio)`. `ratio` is dispatched with `match`, as
  `harmonic_complex` dispatches its `f0`:
  - a number: one ratio for the whole sound;
  - a `(times, ratios)` contour: a ratio that changes over time, read at
    each time window, for a voice whose tract lengthens or shortens;
  - a callable `f -> f_source`: any frequency map, for a warp that is not
    a single ratio (piecewise or bilinear, D3), the inverse map so the
    envelope is read where each new frequency comes from.
  Ratio exactly 1 returns the same data unchanged (C1).
- `Aperiodicity.warp(...)` with the same signature, so moving the
  aperiodicity is one more call when wanted (D6).

Views stay one-way about what they keep: a warp of an envelope is a new
envelope, which `world_synthesize` takes like any other. No new module.

### What these changes are not

They change the vocoder's description of a voice, so they inherit its
limits: minimum-phase pulses (`world.md`), CheapTrick's error rising with
F0 (`female-voices.md`: 0.75 dB at 100 Hz to 3.0 dB at 350 Hz), and D4C's
single band at 16 kHz. A formant shift is a uniform scaling of the whole
envelope, a first-order model of a longer or shorter vocal tract; it does
not change the spectral slope or the voice quality, which C6 shows are a
large part of what separates bdl from slt.

## Decisions

**D1. Where the operations live.**
- *Methods on the views* (recommended): `F0Track.scale`,
  `SpectralEnvelope.warp`, `Aperiodicity.warp`. Each view changes itself,
  the result is the same type, and `world_synthesize` stays the exact port.
  It composes: warp only the envelope, or the envelope and aperiodicity,
  or a time-varying ratio, without a new parameter for each.
- *One function*, `so.change_voice(snd, pitch=1.5, formants=1.2)`, which
  runs the analysis, changes and synthesis, like Praat's "Change gender".
  In its best form it dispatches on its first argument, a `Sound`
  (analyse with defaults) or an already-made `(track, env, ap)` triple
  (skip the slow D4C), and passes `aperiodicity="keep" | "warp"`. One line
  for a stimulus maker; but it hides the three views that `world.md` and
  the aperiodicity page teach, and every analysis option needs a
  pass-through. It could be added later on top of the methods.
- *Keywords on `world_synthesize`* (`formant_ratio=`, `f0_ratio=`): one
  call, and a time-varying ratio could be applied per pulse. But
  `world.md` decided that the WORLD-named functions reproduce WORLD and
  anything different gets its own name.

**D2. A named pitch change.**
- *`F0Track.scale(ratio, range=1.0)`* (recommended): the change is one
  line either way, but a method keeps the `F0Track` (so `.plot` and the
  voicing remain) and the range factor is easy to get wrong by hand
  (applied to unvoiced zeros, or around the mean in Hz).
- *Nothing new*: `(track.t, track.f0 * ratio)` already works with every
  WORLD function. Fewer names; the range factor is left to the reader.
- *Semitones* (`transpose(semitones)`) instead of a ratio: musical, and
  matches `so.pitch_shift(snd, semitones)`; but the formant warp has to be
  a ratio, and two units side by side invite mistakes. A ratio for both is
  recommended; the docstring gives `2 ** (st / 12)`.

**D3. The shape of the warp.**
- *Linear, `f / ratio`* (recommended as the number form): the first-order
  model of a uniformly longer or shorter tract; C3 and C4 show it does
  what it says.
- *Piecewise linear*, the same ratio up to a cut-off and then a line to
  fs/2 that keeps the top fixed: no held values above fs/2 when lowering
  (C4), at the price of a second parameter.
- *Bilinear (all-pass) warping*, as in vocal-tract-length normalisation
  and mel-cepstral analysis: smooth, maps 0 to 0 and fs/2 to fs/2, but its
  parameter is not a formant ratio and it shifts low formants more than
  high ones.
  With the callable form of D1 both alternatives are a function the
  reader passes, so no decision is needed beyond the default.

**D4. Above fs/2 when lowering the formants.**
- *Hold the value at fs/2* (recommended): what the view's interpolation
  already does; at 16 kHz and ratio 0.8 it affects 6.4–8 kHz (C4).
- *Extrapolate the slope* of the top of the envelope: closer to a real
  tract, but a guess, and it can run away.

**D5. The level.**
- *Pure warp* (recommended): the envelope's values move, nothing else; the
  output level changes by −1.5 to +0.4 dB for ratios 0.8 to 1.2 (C4), which
  any `normalize` removes.
- *Keep the power of each time window*: the level stays, but the warp then
  also changes the envelope's values, and "the envelope read at f / ratio"
  is no longer exactly true.

**D6. The aperiodicity.**
- *Keep it as measured; warp it only when asked* (recommended): first-order
  source-filter theory puts the noise in the source, and C5 shows that is
  right where it holds; with D4C at 16 kHz the choice changes the output by
  −28 dB re its level. `Aperiodicity.warp` is there for the other case.
- *Warp it with the envelope by default*: right if the measured noise is
  tied to the resonances, which the harmonic residual's −0.33 on bdl
  hints at (C5); it changes the output by −15 dB re its level with that
  measure. Cho's listening would decide between these better than the
  numbers can.
- *A `change_voice` keyword* (`aperiodicity="keep" | "warp"`) if D1 picks
  the function.

**D7. Ratio 1 is exact.** Return the envelope's data unchanged at ratio 1
(recommended, C1), so identity settings give `world_synthesize`'s output
sample for sample and a test can say so; or accept 1e−16 and test with a
tolerance.

**D8. A listening gallery page, later.** Proposed, not part of this step:
"Changing a voice", with the bdl sentence at F0 × 1.5 (formants kept),
formants × 1.2 (pitch kept), both, the bdl-to-slt numbers of C6 beside the
real slt, the phase vocoder's pitch shift for contrast, and the
aperiodicity kept and warped. Descriptions would say what the analyses
measure, and nothing about how it sounds until Cho has listened.

## Order

1. This document and the checker (this step).
2. After Cho's answers: the methods, tests (identity is exact; the
   measured F0 ratio and fitted warp of C2–C4 on a short synthetic vowel),
   README row, CHANGELOG.
3. The gallery page (D8), as its own PR.

## References

- Boersma, P. & Weenink, D. Praat, "Change gender" (checked through
  praat-parselmouth 0.4.7, Praat 6.1.38, whose error messages name the
  arguments: a pitch floor and ceiling, "Formant shift ratio", "New pitch
  median", "Pitch range factor" and "Duration factor"; its method was not
  read).
- Hillenbrand, J., Getty, L. A., Clark, M. J. & Wheeler, K. (1995).
  Acoustic characteristics of American English vowels. *J. Acoust. Soc.
  Am.* 97, 3099–3111. Table V, as quoted in
  `tools/check_female_voices.py`.
- Morise, M., Yokomori, F. & Ozawa, K. (2016). WORLD: a vocoder-based
  high-quality speech synthesis system for real-time applications. *IEICE
  Trans. Inf. & Syst.* E99-D(7), 1877–1884. doi:10.1587/transinf.2015EDP7457.
- Kominek, J. & Black, A. W. (2004). The CMU Arctic speech databases.
  *Proc. 5th ISCA Speech Synthesis Workshop*, 223–224. The bdl and slt
  recordings (see `docs/speech/SOURCES.md`).
