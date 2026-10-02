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

What was read (all four papers from PDFs Cho supplied):

- **Fant, Liljencrants & Lin (1985)**, *STL-QPSR* 26(4), 1–13 (a paper
  presented at the French–Swedish Symposium, Grenoble, April 22–24, 1985),
  read in full. From it: the open phase (its Eq. 1), the return phase
  (Eq. 11) with ε from Eq. 12, tc set to the period, zero net flow over the
  period, and the return phase as a first-order low-pass with cut-off
  Fa = 1/(2π ta) (Eq. 17), with ΔL = 10 log(1 + f²/Fa²) dB of extra loss.
  Its Eq. 11 has Ee in the return phase; the caption of its Fig. 2 prints
  E0 instead, the slip VOICEBOX's `v_glotlf` notes, and Fant (1986) repeats
  the caption's version. The equations used here are Eqs. 1, 11 and 12.
  **Names differ from the later papers:** in 1985, "Rd" is 2α/ωg (the open
  phase's growth rate) and "Ra" is ta/(tc − te). This document uses Fant's
  (1995) definitions throughout, in which Rd is the shape parameter and
  Ra = ta/T0.
- **Fant (1986)**, *Glottal flow: models and interaction*, read in full: a
  summary of the same model and of the low-pass view of ta (C4).
- **Fant (1995)**, *The LF-model revisited*, was read from the PDF Cho
  supplied (pages 119–126 closely, the rest searched). From it: the
  normalized parameters Ra, Rg, Rk and OQ (its Fig. 1), Rd (Eq. 1), the
  prediction of Ra, Rk and Rg from Rd (Eqs. 2–4), the main range of Rd, the
  OQ, Fa, Rk and Rg it prints for four Rd values (Fig. 5A), typical male
  and female values, and the H1–H2 line (Eq. 8). The formulas used here
  are exactly its Eqs. 2–4; C2 reproduces its Fig. 5A values.
- **Klatt & Klatt (1990)**: read the KLSYN88 source sections (pp.
  838–840), the synthesis-procedure steps for OQ and TL, the summary, the
  parameter table and the abstract; the perception study was read only
  through its abstract and summary. Praat's manual and source code
  (KlattGrid, read in the praat-parselmouth 0.4.7 source distribution) are
  used only for Praat's own tilt filter.

## The LF model in brief [source]

The LF model describes one period of the glottal flow derivative E(t) (the
flow's rate of change, which is what excites the vocal tract once radiation
is folded in). Fant (1986) names its four wave-shape parameters as tp, te,
ta and Ee, with the period ending at tc = T0 = 1/F0:

- **Open phase**, 0 ≤ t ≤ te: E(t) = E0 exp(α t) sin(π t / tp), an
  exponentially growing sinusoid. The flow peaks at tp, where E crosses
  zero, and E reaches its negative peak −Ee at te, the main excitation.
- **Return phase**, te < t ≤ T0: an exponential recovery from −Ee to zero,
  E(t) = −(Ee / (ε ta)) [exp(−ε (t − te)) − exp(−ε (T0 − te))], with
  ε ta = 1 − exp(−ε (T0 − te)). The time constant ta is the projection on
  the time axis of the slope just after te, and says how abrupt the
  closure is: ta = 0 is an instantaneous closure. Fant (1986): in the
  spectrum it is a first-order low-pass with cut-off Fa = 1/(2π ta), above
  which the spectrum falls an extra 6 dB per octave, ΔL = −10 log10(1 +
  (2π ta f)²); ta = 0.15 ms gives Fa = 1060 Hz.
- **Zero net flow**: α is set so that E integrates to zero over the period,
  so the flow starts and ends each cycle at zero.

Fant's normalized parameters, in fractions of the period T0: Ra = ta / T0
(abruptness of closure), Rg = T0 / (2 tp) (the "glottal formant" relative to
F0), Rk = (te − tp) / tp (skew of the pulse). Fant (1995) defines the single
shape parameter

> Rd = (U0 / Ee)(F0 / 110), with U0 / Ee in milliseconds,

that is, Rd = U0 / (0.11 Ee T0), where U0 is the peak flow (the constant
110 makes Rd numerically equal to U0/Ee in ms for an average F0 of 110 Hz).
It gives the main range as 0.3 < Rd < 2.7, from tight, adducted phonation
(small open quotient, high Fa) to breathy, abducted phonation; Rd above 2.7
is for transitions toward full abduction, as at the end of a phrase. Its
Eqs. 2–4 predict the other parameters from Rd alone (subscript p for
predicted):

    Rap = (−1 + 4.8 Rd) / 100                                 (2)
    Rkp = (22.4 + 11.8 Rd) / 100                              (3)
    Rd ≈ (1 / 0.11)(0.5 + 1.2 Rk)(Rk / (4 Rg) + Ra)           (4)

with Rgp from Eq. 4 given Rap and Rkp, which Fant recommends over a
separate regression "to ensure conformity with the LF model". Eq. 4 is an
approximation to Eq. 1: Fant gives its accuracy as 0.5 dB for Rd < 1.4 and
at most 1.7 dB at Rd = 2.7. The open quotient is OQ = (1 + Rk) / (2 Rg).
Typical values he quotes (from Gobl and Karlsson): male vowels Fa = 700 Hz,
Rk = 0.30, Rg = 1.20, which at 100 Hz is close to Rd 0.7; female vowels
Fa = 400 Hz, Rk = 0.30, Rg = 1; "a sonorous voice has a relatively high Fa
of the order of 2000 Hz". And in the flow derivative spectrum, an "almost
perfect linear rise" H1–H2 = −7.6 + 11.1 Rd dB (Eq. 8).

**KLSYN88 and KLGLOTT88** [source]: Klatt & Klatt (1990) gave KLSYN88 a
source switch, SS: 1 = the impulse source of Klatt (1980), the one sonore
has now; 2 = "natural", the new KLGLOTT88 model; 3 = "a slightly modified
version" of the LF model, which they say "can easily be recast" in terms of
AV, F0, OQ (open quotient), SQ (speed quotient) and TL (spectral tilt);
they chose LF because Fant et al. (1985) and Fujisaki and Ljungqvist (1986)
found it better than other models of the same complexity. KLGLOTT88's flow
during the open phase is Ug(t) = at² − bt³, the shape "first proposed by
Rosenberg (1971)", with a and b set by the voicing amplitude and the length
of the open phase, which is OQ in percent of the period (a typical default
50% for a male voice, 60% for a female). The flow derivative 2at − 3bt²
jumps to zero at closure. Its other controls: TL, the extra tilt in dB down
at 3 kHz, applied by a low-pass resonator whose frequency and bandwidth
depend on TL (the mapping is not given in the parts read); FL, a slow
flutter of F0, Δf0 = (FL/50)(F0/100)[sin(2π 12.7 t) + sin(2π 7.1 t) +
sin(2π 4.7 t)] Hz; DI, diplophonic double pulsing; and AH, aspiration.
Praat's KlattGrid implements its own version of TL as a one-pole low-pass
down TL dB at 3 kHz, and generalizes the flow to x^p1 − x^p2.

Their perception study bears on what a listener hears (abstract and
summary): aspiration noise was the perceptually most important cue to
breathiness, and "without its presence, increases to the fundamental
component may induce the sensation of nasality in a high-pitched voice".

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
requested Rd by up to 12%: +0.5 dB at Rd 1 (1.06), +0.3 dB at 1.4,
−1.1 dB at 2.7 (2.38). That is within Fant's own accuracy for Eq. 4 at
large Rd (1.7 dB) and slightly over his 0.5 dB below 1.4 (0.6 dB at Rd 0.5),
so "Rd" is best read as the input to the prediction, not a measured
property. The formulas reproduce the OQ, Fa, Rk and Rg printed in Fant's
(1995) Fig. 5A for Rd 0.3, 0.7, 1.4 and 2.7 to within 2.4% (the rest is his
rounding: Fa 674 Hz against his 660 at Rd 0.7). The open part of the
cycle (te + ta) grows from 36% of the period at Rd 0.3 to 69% at Rd 1 and 91% at Rd 2.7.

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
Fant, Liljencrants & Lin's (1985) first-order low-pass predicts the
change from an abrupt closure (T0 10 ms, tp 4 ms, te 5 ms, 0.5–4 kHz): with ta = 0.15 ms the
spectrum's change follows it to within 0.22 dB of spread, with ta = 0.6 ms
within 1.1 dB, apart from an overall level shift (1.7 and 4.8 dB, because
at fixed Ee a longer return phase carries more flow). The polynomial pulse
(OQ 0.6) falls 6.0 dB/octave: it has no return phase,
which is why KLSYN88 needs a separate tilt filter (Praat's version is down
exactly TL at 3 kHz: 20.0 dB for TL = 20; KLSYN88's is a resonator, not
checked here). At the lips at 100 Hz, from
400 to 3200 Hz, the current RGP source falls 5.8 dB/octave; LF falls 9.6
(Rd 0.5), 10.8 (Rd 0.7), 11.5 (Rd 1) and 12.5 (Rd 2.5).

**C5. H1–H2 follows Rd, and is independent of F0.** [check] The LF flow
derivative's H1–H2 is −4.0 dB at Rd 0.3, 3.8 at Rd 1, 8.3 at Rd 1.4 and
19.3 at Rd 2.7. Fant's Eq. 8 (−7.6 + 11.1 Rd) agrees to within 0.4 dB up
to Rd 1.4, then this computation falls below it: by 1.1 dB at Rd 2 and
3.1 dB at Rd 2.7. Fant calls his line an "almost perfect linear rise" of
his Fig. 6, which this computation does not reproduce above Rd 1.4; the
formulas are his and his Fig. 5A values are reproduced (C2), so the gap is
likely in how his Fig. 6 measured H1 and H2 (from spectra of a sampled
signal, not from the closed form) or in the line fit, but that is a guess.
Across the whole range the closed form follows −6.1 + 9.7 Rd (worst
deviation 0.9 dB).
The current RGP source's H1–H2 is fixed in Hz, not in harmonics: 4.6 dB at
100 Hz and 5.6 dB at 200 Hz. The polynomial pulse's is −2.5 dB at OQ 0.4
and 6.5 dB at OQ 0.7: open quotient is its H1–H2 control.

**C6. The current source is much brighter than any LF voice.** [check] At
the lips at 100 Hz, relative to the first harmonic:

| Harmonic | current RGP | LF Rd 0.5 | LF Rd 0.7 | LF Rd 1 | LF Rd 2.5 |
|---|---|---|---|---|---|
| 2 (200 Hz) | −4.6 dB | +2.0 | −0.3 | −3.8 | −17.6 |
| 10 (1 kHz) | −18.0 | −14.2 | −20.8 | −27.8 | −43.5 |
| 30 (3 kHz) | −27.1 | −30.7 | −38.9 | −46.8 | −64.7 |

Rd 0.7 is close to Fant's typical male values. The current source has the
H1–H2 of Rd 1 but upper harmonics stronger than even Rd 0.5: 12 dB above
the typical male LF pulse at 3 kHz, 20 dB above Rd 1.

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
- **LF Rd 0.7 (typical male, after Fant's quoted values)** and **Rd 1**
  are darker: the upper formants are 10–20 dB weaker relative to F1.
- **LF Rd 0.5 (pressed)**: H2 above H1 and more energy in the middle
  frequencies: the spectrum of a tense voice.
- **LF Rd 2.5 (lax)**: nearly a sinusoid at F0 with weak formants: soft,
  muffled voice. It needs aspiration noise (`AH`) to sound breathy, since
  LF models the periodic pulse only (example 5 adds it). Klatt & Klatt
  (1990) found aspiration the most important breathiness cue, and a
  stronger fundamental alone could sound nasal in a high voice.
- **Phase only** (LF levels in cosine phase): close to LF Rd 1 (C9).
- **An Rd glide** from 0.5 to 2.5 over one second at a fixed pitch: the
  voice relaxes from pressed to lax while the vowel stays /a/; that is the
  didactic point, voice quality as a source property separate from the
  filter.

Apart from that cited finding, these are descriptions of the spectra, not
perception claims; the examples let Cho judge.

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
  - `glottal_source(duration, fs, f0, rd=0.7)`: an LF voiced source on a
    fixed F0 or an F0 contour, built by `harmonic_complex` (C7). `rd` is a
    number or a `(times, values)` track; a track uses the Rd table (C8)
    and needs `harmonic_complex` to accept per-harmonic gains that change
    over time and carry a phase (D4). Like the other generators, RMS 1.
- **stimuli** (`stimuli/klatt.py`): KLSYN88's source switch `SS`, 1 (the
  default) for Klatt's (1980) source, unchanged to the last bit, and 3 for
  LF, as in KLSYN88 (2 stays free for KLGLOTT88, D1); and a new table
  parameter `RD` for the LF shape, used when `SS` is 3. The LF source
  enters as flow through the same radiation difference (C7). Being a table parameter it can be a track, so
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
  abruptness built in, C4, Rd is a single control with a literature behind
  it, and KLSYN88 itself offers LF), LF and KLGLOTT88, or KLGLOTT88 only?
  KLGLOTT88 is simpler (a polynomial) and was KLSYN88's main source, but it
  needs its tilt filter to stop sounding bright, and KLSYN88's tilt
  resonator mapping was not found in the parts read.
- **D2.** Controls: Rd as the one public knob, with `ra`, `rg`, `rk` as an
  expert alternative in `lf_harmonics` only (recommended), or all four
  everywhere, including `klatt_synthesize`?
- **D3.** In `klatt_synthesize`: KLSYN88's switch `SS` (1 = Klatt 1980,
  default; 3 = LF) plus a table parameter `RD` (recommended: it keeps
  Klatt's names as `klatt.md` D5 chose, and every existing call
  bit-for-bit; `RD` can vary in time and interpolates in continua), or
  `RD` alone with 0 meaning Klatt's source (this document's first
  proposal, before Klatt & Klatt was read), or a keyword
  `glottal="klatt"|"lf"`? KLSYN88's own LF takes OQ, SQ and TL rather than
  Rd; Rd is recommended because it is one control (D2).
- **D4.** Time-varying Rd: (a) let `harmonic_complex`'s amplitude function
  return complex gains (level and phase) and take the harmonic number as a
  third argument when it accepts one (recommended: small, and the real-gain
  path is untouched); (b) a fixed Rd per call for now; (c) a separate
  summing loop inside `glottal_source` that does not go through
  `harmonic_complex`.
- **D5.** Public names `lf_harmonics`, `lf_pulse`, `glottal_source`
  (recommended), or fewer: only `glottal_source` public and the rest
  private?
- **D6.** Default Rd: 0.7, close to Fant's typical male values
  (recommended, as `klatt_synthesize`'s defaults are an adult male), or
  1.0, the middle of the range in round numbers? (The Rd formulas
  themselves are now checked against Fant (1995), see C2.)
- **D7.** Phases: the LF pulse's own (recommended: it is the model, and
  `lf_pulse` then matches what is heard), or cosine phase like the current
  source? C9 says the difference is about 1 dB of crest factor.

## Listening examples

`/mnt/project-files/notes/glottal-source/make_examples.py` (project files,
not the repository) writes ten 1 s, 16 kHz files beside itself, all the
same /a/ (formants as in `klatt.md`, F0 130 → 100 Hz): 1 the current
source, 2–4 LF at Rd 0.5, 1 and 2.5, 5 Rd 2.5 with aspiration, 6 LF Rd 1
levels in cosine phase, 7–8 the polynomial pulse at OQ 0.4 and 0.7 (no tilt
filter), 9 an Rd glide from 0.5 to 2.5 at 100 Hz, 10 LF at Rd 0.7. It uses the checker's
prototype for the pulse spectra and the library's `harmonic_complex` and
`resonator` for the rest; its version of the current source matches
`so.klatt_synthesize` to 1e-13, so only the source differs. Not claims.

## References

- Fant, G., Liljencrants, J., & Lin, Q. (1985). A four-parameter model of
  glottal flow. *STL-QPSR* 26(4), 1–13. KTH Speech, Music and Hearing
  (presented at the French–Swedish Symposium, Grenoble, April 1985). Read
  in full.
- Fant, G. (1986). Glottal flow: models and interaction. *Journal of
  Phonetics* 14, 393–399. Read in full.
- Fant, G. (1995). The LF-model revisited. Transformations and frequency
  domain analysis. *STL-QPSR* 36(2–3), 119–156. Read: pp. 119–126, the
  Eq. 8 passage, and the reference list.
- Klatt, D. H. (1980). Software for a cascade/parallel formant synthesizer.
  *JASA* 67(3), 971–995. doi:10.1121/1.383940. (Read for `klatt.md`.)
- Klatt, D. H., & Klatt, L. C. (1990). Analysis, synthesis, and perception
  of voice quality variations among female and male talkers. *JASA* 87(2),
  820–857. doi:10.1121/1.398894. Read in part (see above).
- Rosenberg, A. (1971). Effect of glottal pulse shape on the quality of
  natural vowels. *JASA* 49, 583–590. Not read; cited through Klatt &
  Klatt (1990).
- Brookes, M. VOICEBOX: speech processing toolbox for MATLAB, `v_glotlf.m`
  (version 10865, 2018). Read: code and comments.
- Boersma, P., & Weenink, D. Praat (as packaged in praat-parselmouth
  0.4.7): `KlattGrid.cpp`. Read: the spectral tilt filter.
