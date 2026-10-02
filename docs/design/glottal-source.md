# A glottal pulse model for the formant synthesizer

What it would take to give `so.klatt_synthesize` a glottal source with voice
quality controls: the Liljencrants–Fant (LF) model (Fant, Liljencrants & Lin,
1985), driven by Fant's (1995) single parameter Rd, with the simpler
KLGLOTT88-style polynomial pulse of KLSYN88 (Klatt & Klatt, 1990) weighed as
an alternative. It sets out the model, how it fits the harmonic source
sonore already has, what a listener would hear compared with the current
source, and the decisions for Cho.

Status: proposal. No library code until Cho answers D1–D7.

## Why

`klatt.md` left the glottal pulse shape for later (its D4). The voiced
source today is Klatt's (1980) impulse train through the glottal low-pass
RGP, made from harmonics: one fixed spectrum, falling about 6 dB per octave
at the lips, with nothing to turn. Real voices differ most in the source:
pressed, modal and breathy voice differ in how long the glottis stays open
and how abruptly it closes, which shows up as the level of the first
harmonic against the second (H1–H2) and as the steepness of the spectrum.
A pulse model makes those named, visible controls, which is what the
formant synthesizer is for.

## How the claims are verified, and what was read

As in the other design documents, each claim is numbered and tagged:

- **[source]**: taken from a source listed under "References", with how it
  was read.
- **[check]**: a number printed by `tools/check_glottal_source_claims.py`.
  The script uses only NumPy and SciPy, shares no code with sonore, holds a
  prototype of the LF and polynomial pulses and of Klatt's RGP source, and
  runs in under two seconds. The numbers come from NumPy 2.4.6 and SciPy
  1.17.1.

The sources were harder to reach than usual, so this is said up front:

- **Fant, Liljencrants & Lin (1985)** could not be read: the KTH archive
  returned a server error every time. Its bibliographic details are taken
  from the reference list of VOICEBOX's `v_glotlf` (Brookes), and the
  model's equations from that function's code, which implements them and
  notes that the return-phase equation printed in the paper's Fig. 2 has an
  error. The checker verifies the equations' own properties (zero net flow,
  continuity, the closed-form spectrum), not that they match the paper.
- **Fant (1995)** was read through a web text extractor whose output was
  garbled where equations are. From it: the definition of Rd, its main
  range, and an H1–H2 line. The formulas that predict Ra, Rk and Rg from Rd
  (its Eqs. 2–4) were not legible. The ones used here are the forms widely
  quoted in voice-source code, written from memory; they need checking
  against the paper (D6).
- **Klatt & Klatt (1990)** was not read. What is said about KLGLOTT88 comes
  from Praat's manual and source code (KlattGrid, `PointProcess: To Sound
  (phonation)`), read in the praat-parselmouth 0.4.7 source distribution.

## The LF model in brief [source]

The LF model describes one period of the glottal flow derivative E(t) (the
flow's rate of change, which is what excites the vocal tract once radiation
is folded in) with four timing parameters and one amplitude, as VOICEBOX
implements it:

- **Open phase**, 0 ≤ t ≤ te: E(t) = E0 exp(α t) sin(π t / tp), an
  exponentially growing sinusoid. The flow peaks at tp, where E crosses
  zero, and E reaches its negative peak −Ee at te, the main excitation.
- **Return phase**, te < t ≤ T0: an exponential recovery from −Ee to zero,
  E(t) = −(Ee / (ε ta)) [exp(−ε (t − te)) − exp(−ε (T0 − te))], with
  ε ta = 1 − exp(−ε (T0 − te)). The time constant ta says how abrupt the
  closure is: ta = 0 is an instantaneous closure.
- **Zero net flow**: α is set so that E integrates to zero over the period,
  so the flow starts and ends each cycle at zero.

Fant's normalized parameters, in fractions of the period T0: Ra = ta / T0
(abruptness of closure), Rg = T0 / (2 tp) (the "glottal formant" relative to
F0), Rk = (te − tp) / tp (skew of the pulse). Fant (1995) defines the single
shape parameter

> Rd = (U0 / Ee)(F0 / 110), with U0 / Ee in milliseconds,

that is, Rd = U0 / (0.11 Ee T0), where U0 is the peak flow; it gives the
main range as 0.4 < Rd < 2.7, small Rd being tense (pressed) voice and large
Rd lax, breathy voice. It reports H1–H2 ≈ −7.6 + 11.1 Rd dB. The prediction
formulas used here (D6):

    Ra = (−1 + 4.8 Rd) / 100,   Rk = (22.4 + 11.8 Rd) / 100,
    Rd = (1 / 0.11)(0.5 + 1.2 Rk)(Rk / (4 Rg) + Ra), solved for Rg.

**KLGLOTT88** [source, through Praat]: Praat's manual gives the glottal flow
U(x) = x² − x³ over the open part of the cycle (x running from 0 to 1 while
the glottis is open) as Rosenberg's (1971) shape, "upon which for instance
the Klatt synthesizer is based (Klatt & Klatt, 1990)". The open part's
length is the open quotient OQ; the flow derivative 2x − 3x² ends with a
jump to zero at closure. Praat's KlattGrid adds a one-pole spectral tilt
low-pass, set by how many dB it removes at 3 kHz (TL), and generalizes the
shape to x^p1 − x^p2 (default 3 and 4).

## Claims

**C1. The LF pulse has an exact harmonic spectrum.** [check] Each part of
the LF derivative is an exponential times a sinusoid, so its Fourier
coefficients over one period have a closed form. For Rd 0.3, 1 and 2.7 the
closed form matches numerical integration to 9e-13 (relative), and the net
flow over a cycle is 1e-17. The polynomial pulse's closed form matches to
2e-14. Because the shape is defined in fractions of a period, harmonic k's
coefficient depends only on k and the shape, not on F0.

**C2. Rd sets a feasible pulse, and the pulse's own Rd is close to it.**
[check] Over Rd 0.3–2.7 the predicted Ra, Rk, Rg always give a valid pulse.
The Rd computed back from that pulse's own peak flow differs from the
requested Rd by up to 12% (Rd 1 gives 1.06, Rd 2.5 gives 2.24): Fant's Rd
equation is an approximation, so "Rd" should be read as the input to the
prediction, not a measured property. The open part of the cycle (te + ta)
grows from 36% of the period at Rd 0.3 to 69% at Rd 1 and 91% at Rd 2.7.

**C3. Sampling the pulse aliases; building it from harmonics does not.**
[check] At 200 Hz and 16 kHz (exactly 80 samples per period, so only
aliasing differs), the LF waveform sampled directly differs from its
harmonics below Nyquist by −21 dB (Rd 0.3), −37 dB (Rd 1) and −56 dB
(Rd 2.7) re the signal; the polynomial pulse, which jumps at closure, by
−14 dB. Tense voices alias most because their closure is most abrupt.
VOICEBOX's own code notes this as a known bug of sampling the formula. The
harmonic route has no aliasing and no whole-sample period error (as for
Klatt's impulses, `klatt.md` C4).

**C4. Closure abruptness sets the high-frequency slope.** [check] With a
nearly instantaneous closure (Ra 0.0005) the flow derivative falls
6.1 dB/octave; with Ra 0.01 or 0.05 it falls 12.0 dB/octave above
Fa = F0 / (2π Ra), so Ra is a spectral tilt control built into the pulse.
The polynomial pulse (OQ 0.6) falls 6.0 dB/octave: it has no return phase,
which is why KLSYN88 needs a separate tilt filter (Praat's version is down
exactly TL at 3 kHz: 20.0 dB for TL = 20). At the lips at 100 Hz, from
400 to 3200 Hz, the current RGP source falls 5.8 dB/octave; LF falls 9.6
(Rd 0.5), 11.5 (Rd 1) and 12.5 (Rd 2.5).

**C5. H1–H2 follows Rd, and is independent of F0.** [check] The LF flow
derivative's H1–H2 is −4.0 dB at Rd 0.3, 3.8 at Rd 1 and 19.3 at Rd 2.7,
close to a line −6.1 + 9.7 Rd (worst deviation 0.9 dB). Fant's reported
line gives −4.3, 3.5 and 22.4 at the same Rd; the gap grows at large Rd,
which may come from the prediction formulas (D6) or from the line itself.
The current RGP source's H1–H2 is fixed in Hz, not in harmonics: 4.6 dB at
100 Hz and 5.6 dB at 200 Hz. The polynomial pulse's is −2.5 dB at OQ 0.4
and 6.5 dB at OQ 0.7: open quotient is its H1–H2 control.

**C6. The current source is much brighter than any LF voice.** [check] At
the lips at 100 Hz, relative to the first harmonic:

| Harmonic | current RGP | LF Rd 0.5 | LF Rd 1 | LF Rd 2.5 |
|---|---|---|---|---|
| 2 (200 Hz) | −4.6 dB | +2.0 | −3.8 | −17.6 |
| 10 (1 kHz) | −18.0 | −14.2 | −27.8 | −43.5 |
| 30 (3 kHz) | −27.1 | −30.7 | −46.8 | −64.7 |

The current source has a modal H1–H2 but the high harmonics of a pressed
voice and stronger still: 20 dB above modal LF at 3 kHz.

**C7. A fixed voice quality is a harmonic complex with fixed amplitudes and
phases.** [check] The cosine sum with amplitudes 2|c_k| and phases
arg c_k equals the complex Fourier sum to 2e-15. Since c_k does not depend
on F0 (C1), `harmonic_complex(duration, fs, f0, harmonics, amplitudes,
phases)` with those two arrays already makes an LF source on any F0
contour, each period stretched to its own length, with no change to
`harmonic_complex`. Radiation: the LF E is the true derivative of the flow,
but `klatt_synthesize` takes a first difference, which is 3.1 dB below the
derivative at 0.45 fs; so the LF source should enter as flow coefficients
c_k / (2πik) through the same difference, keeping radiation in one place.

**C8. A voice quality that changes over time is cheap.** [check] Solving
the LF shape for one Rd and its first 80 harmonics takes 0.15 ms. A table
over Rd at 0.002 steps (1,201 shapes, about 0.2 s once), interpolated
linearly in level and unwrapped phase, is within 0.001 dB and 0.0004 rad of
the exact harmonics within 40 dB of the strongest. At 0.01 steps the levels
are still within 0.02 dB but the phases of high harmonics wrap between
steps (errors up to π), so the finer table is needed.

**C9. LF's own phases matter less than its levels.** [check] At Rd 1 and
100 Hz, the source's crest factor is 9.5 dB with LF phases and 10.5 dB with
the same levels in cosine phase (what the current source uses); through
/a/ formants, 7.0 against 7.7 dB. So the audible change comes mostly from
the levels (C5, C6), not the phases.

## What a listener hears

From C4–C6, and from the listening examples below (same /a/, same F0
glide from 130 to 100 Hz, same formants; output levels equalized):

- **The current source** is bright: modal H1–H2 but strong upper
  harmonics, so the upper formants stand out.
- **LF Rd 1 (modal)** is darker: the upper formants are 15–20 dB weaker
  relative to F1.
- **LF Rd 0.5 (pressed)**: H2 above H1 and more energy in the middle
  frequencies: the spectrum of a tense voice.
- **LF Rd 2.5 (lax)**: nearly a sinusoid at F0 with weak formants: soft,
  muffled voice. It needs aspiration noise (`AH`) to sound breathy, since
  LF models the periodic pulse only (example 5 adds it).
- **Phase only** (LF levels in cosine phase): close to LF Rd 1 (C9).
- **An Rd glide** from 0.5 to 2.5 over one second at a fixed pitch: the
  voice relaxes from pressed to lax while the vowel stays /a/; that is the
  didactic point, voice quality as a source property separate from the
  filter.

These are descriptions of the spectra, not perception claims; the examples
let Cho judge.

## Proposed design

Layers follow `layout.md`.

- **signals** (`signals/generators.py`, beside `harmonic_complex`):
  - `lf_harmonics(rd, harmonics)`: complex coefficients of the LF flow
    derivative per harmonic number, from the closed form (C1). With
    `ra=`, `rg=`, `rk=` instead of `rd`, the shape is given directly (the
    literature's names, an allowed exception in `philosophy.md`).
  - `lf_pulse(rd, x)`: one period of the flow derivative (and, with
    `flow=True`, the flow) at fractions of a period `x`, for plotting.
    Sampled, so it aliases (C3); documented as a picture, not a source.
  - `glottal_source(duration, fs, f0, rd=1.0)`: an LF voiced source on a
    fixed F0 or an F0 contour, built by `harmonic_complex` (C7). `rd` is a
    number or a `(times, values)` track; a track uses the Rd table (C8)
    and needs `harmonic_complex` to accept per-harmonic gains that change
    over time and carry a phase (D4). Like the other generators, RMS 1.
- **stimuli** (`stimuli/klatt.py`): a new table parameter `RD`. `RD = 0`
  (the default) is Klatt's RGP source, unchanged to the last bit; `RD > 0`
  is the LF source with that Rd, entering as flow through the same
  radiation difference (C7). Being a table parameter it can be a track, so
  a voice can relax at the end of a phrase, and `klatt_continuum` can make
  an Rd continuum. Levels keep their meaning: the voiced source still has
  RMS 1 at the lips at `AV` = 60.
- **gallery**: a "Voice quality" section on the Formant synthesis page:
  one period of the LF pulse and its flow for three Rd values, their
  harmonic spectra with H1–H2 marked, the same vowel with each, and the Rd
  glide.

`harmonic_complex` with an amplitude function today receives `(t, f)`, the
time and each harmonic's frequency. An LF gain depends on the harmonic
number, which is `f / F0(t)`; and it carries a phase. D4 is the choice of
how to pass both.

### Relation to the WORLD work

WORLD's synthesis (thread on `docs/design/world.md`, landing separately)
excites a measured spectral envelope that already contains the voice
source; it has no separate glottal pulse, so an LF source does not plug
into it. Splitting a WORLD envelope into an LF source and a vocal tract
(inverse filtering, or fitting Rd to a recording) is a different and much
harder problem and is not proposed here. The only shared piece is
`harmonic_complex`, and the change in D4 leaves its existing output
unchanged. This proposal touches none of the WORLD files.

## Order

1. `lf_harmonics` and `lf_pulse`, with tests against the checker's closed
   form and numerical integration (C1), zero net flow, and the Rd table
   (C8).
2. `glottal_source` for a fixed Rd (no change to `harmonic_complex`, C7).
3. The `harmonic_complex` change for time-varying Rd (D4), checked to leave
   every existing output bit-for-bit identical.
4. `RD` in `klatt_synthesize`; `RD = 0` checked bit-for-bit against main.
5. The gallery section.

## Decisions

- **D1.** Which pulse model: LF only (recommended: it has the closure
  abruptness built in, C4, and Rd is a single control with a literature
  behind it), LF and KLGLOTT88, or KLGLOTT88 only? KLGLOTT88 is simpler
  (a polynomial) and what KLSYN88 users know, but it needs a separate tilt
  filter to stop sounding bright, and aliases worst if ever sampled (C3).
- **D2.** Controls: Rd as the one public knob, with `ra`, `rg`, `rk` as an
  expert alternative in `lf_harmonics` only (recommended), or all four
  everywhere, including `klatt_synthesize`?
- **D3.** In `klatt_synthesize`: a table parameter `RD`, 0 = Klatt's
  source (recommended: it can vary in time, interpolates in continua, and
  keeps every existing call bit-for-bit), or a keyword
  `glottal="klatt"|"lf"` with Rd as a separate argument?
- **D4.** Time-varying Rd: (a) let `harmonic_complex`'s amplitude function
  return complex gains (level and phase) and take the harmonic number as a
  third argument when it accepts one (recommended: small, and the real-gain
  path is untouched); (b) a fixed Rd per call for now; (c) a separate
  summing loop inside `glottal_source` that does not go through
  `harmonic_complex`.
- **D5.** Public names `lf_harmonics`, `lf_pulse`, `glottal_source`
  (recommended), or fewer: only `glottal_source` public and the rest
  private?
- **D6.** The Rd prediction formulas could not be read in Fant (1995).
  Could Cho check its Eqs. 2–4 (or upload the PDF) before the library code
  lands (recommended), or accept them as widely quoted, with the docstring
  saying so? They decide every number in C2, C5 and C6 at a given Rd.
- **D7.** Phases: the LF pulse's own (recommended: it is the model, and
  `lf_pulse` then matches what is heard), or cosine phase like the current
  source? C9 says the difference is about 1 dB of crest factor.

## Listening examples

`/mnt/project-files/notes/glottal-source/make_examples.py` (project files,
not the repository) writes nine 1 s, 16 kHz files beside itself, all the
same /a/ (formants as in `klatt.md`, F0 130 → 100 Hz): 1 the current
source, 2–4 LF at Rd 0.5, 1 and 2.5, 5 Rd 2.5 with aspiration, 6 LF Rd 1
levels in cosine phase, 7–8 the polynomial pulse at OQ 0.4 and 0.7 (no tilt
filter), 9 an Rd glide from 0.5 to 2.5 at 100 Hz. It uses the checker's
prototype for the pulse spectra and the library's `harmonic_complex` and
`resonator` for the rest; its version of the current source matches
`so.klatt_synthesize` to 1e-13, so only the source differs. Not claims.

## References

- Fant, G., Liljencrants, J., & Lin, Q. (1985). A four-parameter model of
  glottal flow. *STL-QPSR* 26(4), 1–13. KTH Speech, Music and Hearing.
  Not read (server error); details and equations via VOICEBOX `v_glotlf`.
- Fant, G. (1995). The LF-model revisited. Transformations and frequency
  domain analysis. *STL-QPSR* 36(2–3), 119–156. Read in part, through a
  text extractor (see above).
- Klatt, D. H. (1980). Software for a cascade/parallel formant synthesizer.
  *JASA* 67(3), 971–995. doi:10.1121/1.383940. (Read for `klatt.md`.)
- Klatt, D. H., & Klatt, L. C. (1990). Analysis, synthesis, and perception
  of voice quality variations among female and male talkers. *JASA* 87,
  820–856 (pages as given in Praat's manual). doi:10.1121/1.398894. Not
  read.
- Rosenberg, A. (1971). Effect of glottal pulse shape on the quality of
  natural vowels. *JASA* 49, 583–590 (as given in Praat's manual). Not
  read.
- Brookes, M. VOICEBOX: speech processing toolbox for MATLAB, `v_glotlf.m`
  (version 10865, 2018). Read: code and comments.
- Boersma, P., & Weenink, D. Praat (as packaged in praat-parselmouth
  0.4.7): `KlattGrid.cpp`, `manual_KlattGrid.cpp`, `manual_Fon.cpp`. Read:
  code and manual text.
