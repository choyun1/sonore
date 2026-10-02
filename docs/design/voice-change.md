# Changing a voice: pitch and formants

The design of two changes to a recorded voice that work whatever measured
it and whatever puts it back together:

- a **pitch change**, which multiplies the F0 track and leaves the
  spectral envelope alone, so the formants stay where they were;
- a **formant shift**, which moves the spectral envelope along frequency
  and leaves the F0 track alone, so the pitch stays where it was.

The F0 can come from `so.f0_track`, WORLD's Harvest or `Cepstrum.f0`; the
envelope from CheapTrick or the cepstral lifter; the result can be
synthesized by `so.world_synthesize` or `so.harmonic_complex`. The two
operations act on what all of these share, an F0 contour and an envelope
read as `env(t, f)`, not on any one estimator (Cho, 2026-10-02: "shouldn't
voice warping be method agnostic?").

Together they are the two numbers of the classic "change gender" manipulation
(Praat's command of that name takes a formant shift ratio and a new pitch
median, among others). The question the design has to settle beyond the
two operations is what happens to the aperiodicity when the envelope moves.
Because the operations do not care where their inputs came from, the same
harness also compares the methods: which F0 tracker and which envelope
resynthesize a voice best, unchanged and changed ("Comparing methods").

Status: accepted. Cho accepted every recommendation (D1–D12) on
2026-10-02, and step 2 of "Order" is built: `sonore.analysis.voice`
(`scale_f0`, `warp_frequency`, `GridEnvelope`), `Cepstrum.envelope_view`,
`MFCC.envelope_view`, and `world_synthesize` and `harmonic_complex` taking
any envelope. For D4's open point, `world_synthesize` reads another
envelope on its own grid and its docstring asks for a smooth one (C10),
rather than raising the FFT length. Every number is printed by
`tools/check_voice_change_claims.py` (C1–C7) or
`tools/compare_voice_methods.py` (C8–C11), both rerun on 2026-10-02 after
`so.MFCC` was merged; both prototype the operations rather than call the
library, which `tests/analysis/test_voice.py` checks. The scripts measure what
the library's own analyses see; whether the changed voices sound right is
for Cho to judge by listening, and nothing here claims it.

## What exists already

- `world_synthesize` takes any F0 track, and its docstring already says a
  changed F0 is a pitch shift with the envelope and aperiodicity kept. So a
  pitch change needs no new code: `(track.t, track.f0 * 1.5)`. What is
  missing is a statement of what it does to the measured pitch and
  formants (C2), and a named form if Cho wants one (D5).
- `so.pitch_shift` (phase vocoder) changes pitch by stretching and
  resampling, which moves the formants with the pitch (C2 measures it).
- There is no formant shift anywhere in sonore, and no gallery page or
  section changes a voice with WORLD. `world.md` D9 listed "pitch change,
  formant shift, breathiness" as possible demos; the aperiodicity page did
  not include them.
- **F0 is already method-agnostic.** `harmonic_complex` takes an
  `F0Contour`, anything with `.t` and `.f0`, or a `(times, f0)` pair, so an
  `F0Track`, a Harvest track and `Cepstrum.f0`'s first two outputs all
  work. `world_synthesize` takes the same, but needs time windows evenly
  spaced from time 0, as WORLD assumes; `Cepstrum.f0`'s windows start at
  −15 ms, so it has to be read onto that grid first.
- **Envelopes are not, yet.** `SpectralEnvelope` and `Aperiodicity` are
  callable, `env(t, f)`, giving power on a grid of shape `(n_channels,
  len(f), len(t))`, linear in time and in dB between bins. A linear
  formant shift is that call at `f / ratio`. But `Cepstrum.envelope()`
  returns a bare magnitude array, and `harmonic_complex` reads
  `amplitudes(t, f)` point by point (one amplitude per sample and
  harmonic, `t` and `f` the same shape), not on a grid. So sonore has two
  envelope conventions, and the cepstral envelope has neither.
- **`so.MFCC` (`mfcc.md`) gives an envelope at frequencies only.**
  `mfcc.envelope(f)` is the smoothed band power the kept coefficients
  imply (mel bands and a truncated cosine transform discard detail),
  offered for display, on the MFCC's own time windows. It enters here as
  one more envelope source once it is read as `env(t, f)` (D3).

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

C7 mixes sources: F0 from `so.f0_track`, Harvest (the stored track
`docs/speech/bdl_arctic_a0131_f0.csv`) and `Cepstrum.f0` (40 ms Hann time
windows, hop 5 ms); envelope from CheapTrick or the cepstral lifter at half
the median voiced period; synthesis by `world_synthesize` (any envelope
sampled at WORLD's time windows and frequencies) or `harmonic_complex`
(the envelope sampled once on a 5 ms by 10 Hz grid and read point by
point, as the Voices from harmonics gallery page does). Each changed output
is compared with the same sources synthesized unchanged.

A second script, `tools/compare_voice_methods.py`, compares the methods
against each other ("Comparing methods" below); it runs in about five
minutes, and uses pyworld 0.3.5 (development only, as in
`crosscheck_world_vocoder.py`) for Harvest, which it skips if pyworld is
missing.

The checker runs in about a minute (NumPy 2.4.6, SciPy 1.17.1).

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

**C7. Both operations do the same thing whatever measured the voice and
whatever synthesizes it.** Pitch × 1.5 and formants × 1.2 on the bdl
sentence, against the same sources unchanged:

| F0 from | envelope | synthesizer | F0 out/unchanged | within 5% | fitted warp (residual) |
|---|---|---|---|---|---|
| `so.f0_track` | CheapTrick | `world_synthesize` | 1.500 | 100% | 1.200 (0.36 dB) |
| `so.f0_track` | cepstral | `world_synthesize` | 1.501 | 100% | 1.199 (0.62 dB) |
| Harvest | CheapTrick | `world_synthesize` | 1.500 | 98% | 1.200 (0.35 dB) |
| Harvest | CheapTrick | `harmonic_complex` | 1.500 | 100% | 1.201 (0.27 dB) |
| Harvest | cepstral | `world_synthesize` | 1.500 | 100% | 1.200 (0.57 dB) |
| Harvest | cepstral | `harmonic_complex` | 1.500 | 100% | 1.200 (0.55 dB) |
| `Cepstrum.f0` | CheapTrick | `world_synthesize` | 1.500 | 100% | 1.200 (0.34 dB) |
| `Cepstrum.f0` | cepstral | `world_synthesize` | 1.500 | 100% | 1.199 (0.57 dB) |

`world_synthesize` uses D4C measured with the same F0; `harmonic_complex`
has no noise (unvoiced stretches silent). The prototypes need only the
interface: a warped envelope is any envelope read at `f / ratio`, and the
synthesizers see an envelope like any other. Sources differ in what the
output sounds like, which these numbers do not judge. [check]

## Comparing methods

Because the operations take any F0 contour and any envelope, the same
harness can ask which tracker and which envelope resynthesize a voice
best, unchanged and changed (Cho, 2026-10-02: "mixing and matching F0 and
envelope methods, see how well each one works at resynthesizing or voice
warping"). `tools/compare_voice_methods.py` does this. The claims below
are tagged [compare].

**Sources.** F0: the truth (synthetic vowels only), `so.f0_track`,
Harvest, `Cepstrum.f0`. Envelope: the truth (synthetic only), CheapTrick,
the cepstral lifter, and `so.MFCC`'s envelope (13 coefficients, 26 HTK
bands, 25 ms symmetric Hamming, 10 ms hop), with its default height-1
triangles and with area-normalized ones (`triangles="area"`). Synthesis: `world_synthesize` (with
D4C measured on the same F0) and `harmonic_complex` (no noise).

**Scores.** On synthetic vowels the truth is exact, so the score uses none
of the methods compared: the output's harmonics are measured by least
squares at their known frequencies over 0.2–0.6 s and compared, level
removed, with the true envelope over 100–5000 Hz (after a change, with the
true envelope read at `f / 1.2` at harmonics of 1.5 F0). Twelve vowels:
heed, hod and who'd with men's formants at F0 110 and 150 Hz and with
women's at 200 and 250 Hz. On recordings there is no truth; the score is
the level-free distance between the original's and the output's 40-band
log mel spectra (median over voiced time windows), which smooths the way
the MFCC envelope does and so favours it. It is a sanity check, not a
ranking.

**C8. The envelope decides the result; the F0 source and the synthesizer
barely matter on steady vowels.** [compare] Median (worst) error over the
twelve vowels, true F0, `world_synthesize`:

| envelope | resynthesis | pitch × 1.5, formants × 1.2 |
|---|---|---|
| CheapTrick | 0.66 dB (0.86) | 1.11 dB (2.18) |
| cepstral | 2.62 dB (3.37) | 2.18 dB (3.32) |
| MFCC (13), height-1 triangles | 4.35 dB (7.46) | 4.18 dB (5.51) |
| MFCC (13), area-normalized | 3.23 dB (5.98) | 3.32 dB (5.22) |

With `harmonic_complex` the same envelopes give 0.66, 2.73, 4.34 and 3.23
dB (within 0.11 dB of WORLD's), and the truth gives 0.00 dB, so the score
itself adds nothing. Swapping the true F0 for Harvest or `Cepstrum.f0`
changes the CheapTrick row by at most 0.02 dB; their F0 errors are 0.02%
and 0.04% (median). By voice, CheapTrick's resynthesis error is 0.51 dB on
the men's vowels and 0.80 dB on the women's; the cepstral envelope's 1.56
and 3.20 dB; the MFCC envelope's 4.94 and 3.85 dB (area-normalized: 3.75
and 2.90 dB). Area-normalized triangles help because `mfcc.envelope` gives
band powers, sums over triangles that widen with frequency, so with
height-1 triangles the envelope tilts upward against a spectral density;
the remaining error is what 26 bands and 13 coefficients smooth away
(`mfcc.md`, C4).

**C9. A wobbling F0 costs more than its median error suggests.** [compare]
`so.f0_track` has a median F0 error of 0.08%, but on women's heed at 250
Hz its estimate wanders between 247 and 252 Hz, and the resynthesis error
with CheapTrick rises to 3.76 dB (Harvest: 0.86 dB). The output's upper
harmonics wander with it, up to 50 Hz at 5 kHz, which the fixed-frequency
score reads as a loss; a listener might hear it as roughness or not at
all, which only listening can say. On these steady vowels the trackers
differ in nothing else; their voicing errors on real speech are in
`f0.md` and `female-voices.md`.

**C10. WORLD's synthesis needs a smooth envelope: given the exact one it
does worse than given CheapTrick's estimate.** [compare] With the true
envelope, `world_synthesize`'s resynthesis error is 2.01 dB median and
7.03 dB worst (men's who'd at 150 Hz), against 0.66 and 0.86 dB with
CheapTrick. For that vowel, whose envelope spans 95 dB over 50–5000 Hz,
the error is 7.04 dB at WORLD's FFT size (1024 at 16 kHz), 2.08 dB at 4096
and 1.86 dB at 16384, so most of it comes from computing the
minimum-phase response of a deep-valleyed envelope on too few points.
CheapTrick's estimate is within 0.74 dB of the truth at the harmonics and
much smoother between them. This matters for method-agnostic synthesis: an
envelope from another source (a formant model, a hand-drawn one) may need
a larger FFT or smoothing before WORLD's synthesis, which D4 has to
handle.

**C11. On the recordings CheapTrick still resynthesizes best, even under a
score that favours the MFCC envelope.** [compare] Level-free log mel
distance, median over voiced time windows:

| speaker | F0 from | CheapTrick | cepstral | MFCC (13) | MFCC, area |
|---|---|---|---|---|---|
| bdl | `so.f0_track` | 1.70 dB | 3.02 dB | 4.65 dB | 3.18 dB |
| bdl | Harvest | 1.58 dB | 2.95 dB | 4.65 dB | 3.12 dB |
| bdl | `Cepstrum.f0` | 1.79 dB | 3.18 dB | 4.78 dB | 3.27 dB |
| slt | `so.f0_track` | 1.97 dB | 5.09 dB | 5.14 dB | 3.58 dB |
| slt | Harvest | 1.97 dB | 5.16 dB | 5.12 dB | 3.53 dB |
| slt | `Cepstrum.f0` | 2.27 dB | 5.19 dB | 5.25 dB | 3.69 dB |

The cepstral envelope loses most on the higher voice (slt), as
`female-voices.md` found on synthetic vowels; on slt the area-normalized
MFCC envelope does better than the cepstral one (3.5–3.7 dB against
5.1–5.2), on bdl about the same. There is no ground truth for
a voice change on a recording, so changes are compared on the synthetic
vowels only.

**What this does not say.** Every score is a spectral distance. None says
how natural, or how much like the target, a voice sounds; that is Cho's
listening, and a gallery page (D11) would put the mixes side by side to
hear. Steady synthetic vowels have no consonants, onsets or voicing
changes, where the F0 trackers differ most.

## Proposed design

Two functions on two interfaces, both of which sonore already half has:

- an **F0 contour**: anything with `.t` and `.f0`, or a `(times, f0)`
  pair (the existing `F0Contour`);
- an **envelope**: anything callable as `env(t, f)` returning power,
  shape `(n_channels, len(f), len(t))` (the existing `SpectralEnvelope`
  convention; `Aperiodicity` reads the same way).

```python
track = so.f0_track(snd)               # or a Harvest pair, or Cepstrum.f0
env = so.cheaptrick(snd, track)        # or cepstrum.lifter(...).envelope_view()
ap = so.d4c(snd, track)

higher = so.world_synthesize(so.scale_f0(track, 1.5), env, ap)            # pitch only
shorter_tract = so.world_synthesize(track, so.warp_frequency(env, 1.2), ap)  # formants only
on_harmonics = so.harmonic_complex(snd.duration, snd.fs, so.scale_f0(track, 1.53),
                                   amplitudes=so.warp_frequency(env, 1.17))
```

- `so.scale_f0(contour, ratio, *, range=1.0)` multiplies the voiced
  values by `ratio` and, if `range` is not 1, spreads them around their
  median on a log scale: `median * ratio * (f0 / median) ** range`.
  (Praat's "Change gender" has a "pitch range factor"; this definition is
  sonore's, not read from Praat.) Unvoiced windows stay 0. It returns what
  it was given: an `F0Track` stays an `F0Track` (so `.plot` and the
  voicing stay), a pair stays a pair.
- `so.warp_frequency(view, ratio)` returns a view read as
  `new(t, f) = view(t, f / ratio)`. It never looks inside the view, so it
  moves a CheapTrick envelope, a cepstral one, an aperiodicity, or
  anything else read as `(t, f)`. `ratio` is dispatched with `match`, as
  `harmonic_complex` dispatches its `f0`:
  - a number: one ratio for the whole sound;
  - a `(times, ratios)` contour: a ratio that changes over time, for a
    voice whose tract lengthens or shortens;
  - a callable `f -> f_source`: any frequency map (piecewise or bilinear,
    D6), giving where each new frequency is read from.
  Ratio exactly 1 returns the view itself (C1).
- Every envelope source offers the interface. `SpectralEnvelope` and
  `Aperiodicity` already do; `Cepstrum` gains a view of its lifted
  envelope (D3), on the general grid class below.
- Every synthesizer takes any envelope (D4). `world_synthesize` uses its
  own `SpectralEnvelope` as it is (so the exact port is untouched) and
  samples any other envelope at its time windows and frequencies, which
  it already knows from the aperiodicity; it reads an F0 contour that is
  not on its grid onto it. `harmonic_complex` recognises a grid envelope
  and reads it point by point itself, as amplitude.

### What these changes are not

They change a description of a voice, so they inherit the limits of
whichever analysis and synthesis are used; with WORLD's: minimum-phase pulses (`world.md`), CheapTrick's error rising with
F0 (`female-voices.md`: 0.75 dB at 100 Hz to 3.0 dB at 350 Hz), and D4C's
single band at 16 kHz. A formant shift is a uniform scaling of the whole
envelope, a first-order model of a longer or shorter vocal tract; it does
not change the spectral slope or the voice quality, which C6 shows are a
large part of what separates bdl from slt.

## Decisions

**D1. Where the operations live.**
- *Free functions on the interfaces* (recommended): `so.scale_f0`,
  `so.warp_frequency`. They depend on nothing but the interface, so any
  F0 tracker, any envelope estimator and any synthesizer combine (C7), and
  a new estimator joins by offering `env(t, f)`, without the operations
  changing. `world_synthesize` stays the exact port.
- *Methods on each class* (`F0Track.scale`, `SpectralEnvelope.warp`,
  `Aperiodicity.warp`): reads well in a chain, but a Harvest pair or a
  cepstral envelope has no such method, which is the objection Cho raised.
  At its best, each method is a one-line call to the free function, so
  they could be added later as sugar without a second implementation.
- *One function*, `so.change_voice(snd, pitch=1.5, formants=1.2)`, which
  runs analysis, changes and synthesis, like Praat's "Change gender". In
  its best form it takes `f0=`, `envelope=` and `synthesizer=` as
  callables or ready-made views, defaulting to WORLD's, so it is
  method-agnostic too. One line for a stimulus maker; but it hides the
  views the gallery teaches, and every option needs a pass-through. It
  can sit on top of the free functions later.
- *Keywords on `world_synthesize`*: ties the change to one synthesizer,
  and `world.md` decided the WORLD-named functions reproduce WORLD.

**D2. The envelope convention.**
- *Grid, power: `env(t, f)` gives `(n_channels, len(f), len(t))`*
  (recommended): what `SpectralEnvelope` and `Aperiodicity` already do, so
  nothing merged changes; a grid is what plots and WORLD's synthesis need.
- *Point by point, amplitude*, as `harmonic_complex` reads `amplitudes(t,
  f)`: natural for oscillators, which need one value per sample and
  harmonic, but it would change both merged views.
- Either way, the other side gets an adapter, not a second convention: in
  the recommendation, `harmonic_complex` recognises a grid envelope (D4).

**D3. A general grid envelope.**
- *`so.GridEnvelope(power, t, f)`* (recommended; name open): power on any
  time and frequency grid, read as `env(t, f)` with the same interpolation
  as `SpectralEnvelope`, which becomes the case whose grid is WORLD's.
  `Cepstrum` returns one from `envelope_view()` (squared magnitude, its own
  time windows and frequencies), and any future source (LPC, a
  hand-drawn envelope) can too. `so.MFCC.envelope(f)` takes frequencies
  only; it joins as `so.GridEnvelope(mfcc.envelope(f), mfcc.t, f)`, as
  `compare_voice_methods.py` does, or `MFCC` could gain an
  `envelope_view()` like `Cepstrum`'s. C7's prototype is this class.
- *Leave it to the reader*: wrap the array in a function by hand, as the
  checker does. No new name, but every source needs its own wrapper, and
  dB interpolation is easy to get wrong.

**D4. Synthesizers that take any envelope.**
- *Each synthesizer adapts on the way in* (recommended):
  `world_synthesize` dispatches on its `envelope` (its own
  `SpectralEnvelope` on its grid: used as it is, exact; anything else:
  sampled at its time windows and frequencies) and reads off-grid F0
  contours onto its time windows; `harmonic_complex` dispatches on
  `amplitudes` (an array, a point-by-point function, or a grid envelope
  it reads point by point as amplitude). Because WORLD's synthesis
  distorts a deep-valleyed envelope at its own FFT size (C10), the
  sampling step either uses a larger FFT when the envelope is not
  WORLD's own, or says in the docstring that a smooth envelope is
  expected; which one is part of this decision. The adaptations are listed in
  `DIFFERENCES_FROM_WORLD` only if they change an output WORLD could give,
  which they do not.
- *An explicit conversion*, `so.to_world_grid(env, aperiodicity)` and
  `env.amplitude` for `harmonic_complex`: nothing implicit, at the cost of
  a step the reader must know about for each synthesizer.

**D5. A named pitch change.**
- *`so.scale_f0(contour, ratio, range=1.0)`* (recommended): the change is
  one line either way, but the function keeps the contour's type and the
  range factor is easy to get wrong by hand (applied to unvoiced zeros, or
  around the mean in Hz).
- *Nothing new*: `(track.t, track.f0 * ratio)` already works with every
  WORLD function. Fewer names; the range factor is left to the reader.
- *Semitones* (`transpose(semitones)`) instead of a ratio: musical, and
  matches `so.pitch_shift(snd, semitones)`; but the formant warp has to be
  a ratio, and two units side by side invite mistakes. A ratio for both is
  recommended; the docstring gives `2 ** (st / 12)`.

**D6. The shape of the warp.**
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
  With the callable form of `ratio` both alternatives are a function the
  reader passes, so no decision is needed beyond the default.

**D7. Above fs/2 when lowering the formants.**
- *Hold the value at fs/2* (recommended): what the view's interpolation
  already does; at 16 kHz and ratio 0.8 it affects 6.4–8 kHz (C4).
- *Extrapolate the slope* of the top of the envelope: closer to a real
  tract, but a guess, and it can run away.

**D8. The level.**
- *Pure warp* (recommended): the envelope's values move, nothing else; the
  output level changes by −1.5 to +0.4 dB for ratios 0.8 to 1.2 (C4), which
  any `normalize` removes.
- *Keep the power of each time window*: the level stays, but the warp then
  also changes the envelope's values, and "the envelope read at f / ratio"
  is no longer exactly true.

**D9. The aperiodicity.**
- *Keep it as measured; warp it only when asked* (recommended): first-order
  source-filter theory puts the noise in the source, and C5 shows that is
  right where it holds; with D4C at 16 kHz the choice changes the output by
  −28 dB re its level. `so.warp_frequency(ap, ratio)` is there for the other case.
- *Warp it with the envelope by default*: right if the measured noise is
  tied to the resonances, which the harmonic residual's −0.33 on bdl
  hints at (C5); it changes the output by −15 dB re its level with that
  measure. Cho's listening would decide between these better than the
  numbers can.
- *A `change_voice` keyword* (`aperiodicity="keep" | "warp"`) if D1 adds
  the function.

**D10. Ratio 1 is exact.** Return the view itself at ratio 1
(recommended, C1), so identity settings give `world_synthesize`'s output
sample for sample and a test can say so; or accept 1e−16 and test with a
tolerance.

**D11. A listening gallery page, later.** Proposed, not part of this step:
"Changing a voice", with the method comparison of C8–C11 as sounds to
hear side by side, the bdl sentence at F0 × 1.5 (formants kept),
formants × 1.2 (pitch kept), both, the bdl-to-slt numbers of C6 beside the
real slt, the phase vocoder's pitch shift for contrast, and the
aperiodicity kept and warped. Descriptions would say what the analyses
measure, and nothing about how it sounds until Cho has listened.

**D12. Where the comparison lives.**
- *A script in `tools/`* (recommended): `compare_voice_methods.py` as
  here, rerun when a method is added. The scores are choices (which range, which smoothing),
  and a script states them where they are made.
- *A library function* (`so.compare_resynthesis(sources, ...)`): reusable
  on a reader's own recordings; but it would make one scoring rule look
  official, and the sources are already one line each with the proposed
  interfaces.
- *Part of the gallery page* (D11): tables and the sounds side by side,
  regenerated with the page. The script would still be where the numbers
  come from.

## Order

1. This document and the checker (this step).
2. After Cho's answers: the functions, the grid envelope and the
   synthesizers' dispatch, tests (identity is exact; the
   measured F0 ratio and fitted warp of C2–C4 on a short synthetic vowel),
   README row, CHANGELOG.
3. The gallery page (D11), as its own PR.

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
