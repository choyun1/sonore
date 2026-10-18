# The Voices group, rebuilt around the source-filter model

Status: proposed, not built. Cho asked for this on 2026-10-18, while the WORLD ports were being
renamed (PR #238), and asked to see the design before anything is implemented. The decisions at
the end need Cho's answer first.

Cho's asks, in order:

- Drop WORLD's synthesis. Matching WORLD sample for sample doesn't matter, because sonore doesn't
  do WORLD's whole analysis and synthesis, so it shouldn't keep a separate WORLD-style synthesizer.
  The voice is rebuilt by a recipe made from the analyses (an F0 track, a pitch-adaptive envelope
  and an aperiodicity) and the synthesizers sonore already has.
- Add a page that explains pitch-adaptive envelopes in prose and with the math, as STRAIGHT and
  WORLD compute them.
- Start the group with a page that explains the source-filter model.
- Cover how voices differ: male and female talkers, and the other indicators of voice quality.
- Move, condense or merge the examples this affects.

The rules from the gallery rewrite (`gallery-rewrite.md`) still hold. Each repeated topic has one
owner page and the other pages link to it, and the two talkers are paired rather than the female
talker coming last. An example that moves keeps its key, so old links still land on it (the
`moved` redirect in `docs/gallery/build.py`).

## What is there now

Seven pages, in this menu order: Two talkers, Formant synthesis, Finding formants, Pitch tracking,
Spectral envelope, Source and aperiodicity, Rebuilding and changing a voice. Most pages explain
the source and the filter in passing, each in its own words:

- Formant synthesis builds a vowel "piece by piece" (fv1 to fv4) and draws source times filter
  (fs1).
- Two talkers ends with "Harmonics sample the envelope" (tk5).
- Spectral envelope splits a recording into envelope and harmonics (c3 to c5).
- Source and aperiodicity opens with buzz, hiss and both (ap1 to ap3).
- Rebuilding and changing a voice opens by putting the envelope back on harmonics (he1 to he3),
  then fills the unvoiced gaps (hu1, hu2, hm4).

Voice quality, the LF pulse at three values of Rd and breathy voice (fq1 to fq5, fn3), sits at the
end of Formant synthesis. The ways the two talkers differ are spread across Two talkers (pitch,
envelopes, long-term spectra), the female-voice findings in `views/female-voices.md`, and "Toward
another talker" (vc14, vc15, vc19) at the end of the voice-change page.

WORLD appears in two places. Source and aperiodicity has a section "What WORLD reports" and ends
with a WORLD resynthesis (ap10). Rebuilding and changing a voice has a section "Resynthesis with
WORLD" (vc2), and every change after it (vc3 to vc19) is synthesized by
`so.source_filter_synthesize`.

## Proposed pages

Nine pages, two of them new. The order runs from the idea, to synthesis by rule, to measuring each
part, to rebuilding a voice:

| # | Page | New or changed |
|---|---|---|
| 1 | The source-filter model | new |
| 2 | Formant synthesis | slimmer |
| 3 | How voices differ (was Two talkers) | renamed and grown |
| 4 | Pitch tracking | unchanged |
| 5 | Finding formants | unchanged |
| 6 | Spectral envelope | slimmer |
| 7 | Pitch-adaptive envelopes | new |
| 8 | Source and aperiodicity | WORLD section reworded, resynthesis by the recipe |
| 9 | Rebuilding and changing a voice | condensed, resynthesis by the recipe |

### 1. The source-filter model (new)

The page every other Voices page leans on. In prose and math:

- A voice is a source through a filter, `s = e * h`, or `S(f) = E(f) H(f)` in spectra. The source
  is the larynx or a constriction, and the filter is the vocal tract from there to the lips.
- A voiced source is a train of pulses one period `T0 = 1/F0` apart, so its spectrum has lines at
  the harmonics `k F0`. The output's harmonics sample the filter: you hear `|H|` only at multiples
  of F0.
- A noise source has a continuous spectrum, so it shows the whole filter. Real voices mix the two,
  and the share of noise at each frequency is the aperiodicity.
- Changing the source changes the pitch and voice quality. Changing the filter changes the vowel
  and the talker's size. Analysis tries to undo the product: an F0 track for the source's
  spacing, an envelope for `|H|`, an aperiodicity for the mix.

Examples moved in, with their keys kept: fv1 to fv4 (a vowel piece by piece), fs1 (source plus
filter), tk5 (harmonics sample the envelope), and ap1 to ap3 (buzz, hiss, both). One new figure
draws `|E|`, `|H|` and their product for a pulse train and for noise. The page ends with a map of
the group: which page measures which part, and which page puts the parts back together.

### 2. Formant synthesis (slimmer)

It keeps the Klatt vowel (fv5), six vowels, consonants from transitions, noise (fn1, fn2),
cascade and parallel, and "sonore says its name". The opening "A vowel, piece by piece" and
"Source times filter" move to page 1, so the page starts from `so.klatt_synthesize`. "Voice
quality" (fq1 to fq5) and the breathy vowel (fn3) move to page 3.

### 3. How voices differ (renamed from Two talkers, grown)

The page that answers "what makes these two voices different?", in the source-filter terms of
page 1. It keeps the sentence read by both talkers, their pitch tracks, envelopes and long-term
spectra (tk1 to tk4). It adds:

- **Source:** F0 level and range for each talker. Voice quality, meaning the LF pulse at three Rd
  values and relaxing into breathy voice (fq1 to fq5, fn3, moved from page 2).
- **Indicators measured on both talkers**, each defined in the text: the harmonic levels H1–H2
  and H1–A3 (spectral tilt), the noise share from `so.band_aperiodicity` in a few bands
  (breathiness), and a vocal-tract length ratio from the formants (`so.formant_track`) as a
  formant scaling factor. None of these numbers has been measured yet. They are computed when the
  page is built, and the page says what each one leaves out.
- **Toward another talker:** a short section that hears each talker at the other's pitch and
  formants. It links to page 9, which owns the voice changes (vc14, vc15, vc19 stay there).

Two talkers is the data page the speech pages share ("measured once"), so its loading code stays
first on the page.

### 4 and 5. Pitch tracking, Finding formants (unchanged)

They only get links back to page 1 and page 3.

### 6. Spectral envelope (slimmer)

It keeps one time window, splitting the voice in two (c3 to c5), and MFCCs (m1 to m3), with the
cepstral lifter as the plain way to read an envelope. The lifter-versus-CheapTrick comparison in
"One time window" moves to page 7. The page ends where the lifter fails, at high F0, which leads
into page 7.

### 7. Pitch-adaptive envelopes (new)

Cho's "higher-quality voice" page, in prose and math. The argument:

- **The problem.** A periodic signal under a fixed window leaves a ripple at F0 in the spectrum
  (a short window) or in time (a long window). The envelope we want is `|H|`, but every measured
  spectrum is `|H|` times the harmonic comb, smeared by the window.
- **STRAIGHT** (Kawahara, Masuda-Katsuse & de Cheveigné, 1999): a window sized to the pitch
  period, smoothing in frequency over F0 with a kernel that keeps the harmonic peaks' level, and a
  complementary window that fills the dips in time. **TANDEM-STRAIGHT** (Kawahara et al., 2008):
  two power spectra half a period apart, added, so the fluctuation over time cancels. Prose and
  equations only, from the papers. sonore has no STRAIGHT code and the plan writes none (D4).
- **CheapTrick** (Morise, 2015), as `so.pitch_adaptive_envelope`: a Hann window three periods
  long, whose power is nearly constant over time, so no second window is needed. Then a moving
  average over `2 F0 / 3`, and liftering by `sinc(F0 q)` times the recovery lifter
  `(1 - 2 q1) + 2 q1 cos(2 pi F0 q)`. Each step gets its formula and a figure, written from the
  function's own docstring.
- **Examples.** The lifter against CheapTrick on one time window, moved from page 6. A window
  length sweep showing ripple trading between frequency and time. Both talkers' envelopes, where
  the higher voice shows the difference most. The effect of `q1`. `TVGaborFrame.pitch_adaptive`
  is linked as the same idea seen as a frame (an invertible analysis rather than an envelope).

### 8. Source and aperiodicity (WORLD wording dropped, resynthesis by the recipe)

Buzz, hiss and both (ap1 to ap3) move to page 1, and this page links back to them. It keeps
"Fit the harmonics, keep the rest" (ap4 to ap6). "What WORLD reports" becomes "What D4C reports"
(ap7). The sentence (ap8) and the listening examples (ap9 to ap13) stay, but are synthesized by
the recipe.

### 9. Rebuilding and changing a voice (condensed)

The page already builds most of the recipe step by step: harmonics shaped by the envelope (he1),
shaped noise in the unvoiced gaps (hu2), and phases on a moving pitch (hf0 to hf3). It gains one
step, the aperiodicity's split of each frequency between harmonics and noise, so that "Resynthesis
with WORLD" becomes "Harmonics plus shaped noise". vc2 is then the recipe's resynthesis, and every
change after it uses the recipe. The phase-vocoder contrast (vc7), pitch range, the aperiodicity
kept and warped, toward another talker, and "Any pitch track, any envelope" stay. The noise-band
examples he2 and he3 (channel-vocoder style) move to the Vocoder page, which owns that idea.

## The recipe that replaces WORLD's synthesis

For each time and frequency, with envelope `S`, noise share `a` (`Aperiodicity.share`) and F0:

- **Harmonics:** `so.harmonic_complex(duration, fs, f0, amplitudes=...)`, with harmonic `n`'s
  amplitude chosen so its power is `S (1 - a)` spread over one harmonic spacing. With a power
  spectral density `P`, a harmonic every `F0` carries `a_n^2 / 2 = 2 F0 P / fs`. The derivation
  goes in the page text.
- **Noise:** white noise, its STFT multiplied by `sqrt(S a)` (by `sqrt(S)` where unvoiced), and
  inverted with `to_sound()`.
- **Output:** the two added together.

Everything in it already exists in sonore. What it loses against the port is the minimum-phase
pulse shape, since the harmonics take whatever phases they're given, and sample-for-sample
agreement with WORLD.

What we know about the quality, and what not:

- **Measured:** on the steady test vowels, with the same envelope, `harmonic_complex` and the port
  come within 0.11 dB of each other in level-free harmonic error (`views/voice-change.md`, C8).
- **Not measured:** that comparison had no noise in the resynthesis. The noise half, its level
  calibration, and the score on the two recordings come before any page is rebuilt. A checker,
  `tools/check_voice_recipe_claims.py`, will report the recipe against the port on the synthetic
  vowels (harmonic error, recovered noise share against the truth) and on the recordings
  (level-free log mel distance, as in C11).

## What leaves the library

`so.source_filter_synthesize` and its tests go. So do the synthesized arrays in
`tests/data/world_reference.npz` and the synthesis parts of `tools/crosscheck_world_vocoder.py`
and `tools/check_world_claims.py`. The two DIFFERENCES_FROM_WORLD items about synthesis (the `rng`
option and the refusal of far-apart pulses) also go. The design notes `views/world.md` and
`views/voice-change.md` get a dated note, as the rename did.

The rest of `views/world.py` stays where it is. It becomes the home of the pieces the two
analysis ports share: WORLD's noise generator (`world_randn`, which D4C uses), its FFT sizes
(`world_fft_size`, `_periods_fft_size`), MATLAB's rounding, the reading of an F0 track onto time
windows, and DIFFERENCES_FROM_WORLD. Its module docstring changes to say that. Nothing that
imports from it has to move (D5).

The analysis ports, `so.pitch_adaptive_envelope` and `so.band_aperiodicity`, still match WORLD to
floating-point precision, tested against stored pyworld output. The plan keeps that (D6). The
reason to drop the synthesis was that its exactness bought nothing. The analyses' exactness is
what lets sonore's envelope and aperiodicity be compared with any paper that uses WORLD's, and it
is already built and tested.

The rest of the library outside `views/world.py` changes only in docstrings. The `back_to_sound`
lines of the envelope, aperiodicity, F0, LPC, cepstrum and MFCC views now point to
`so.source_filter_synthesize`. They will point to `so.harmonic_complex` and the recipe's gallery
section instead.

## Order

1. PR #238 (renames and WORLD's license) merges as it is. It stands on its own.
2. The recipe and its checker, `tools/check_voice_recipe_claims.py`, with the numbers added to
   this note. If the noise half measures badly, the plan comes back to Cho before step 3.
3. Remove `so.source_filter_synthesize`, and move pages 8 and 9 to the recipe (one PR).
4. Pages 1 and 7, the new pages, with the examples that move into them (one PR each).
5. Page 3, then the slimmer pages 2 and 6 and the moves to the Vocoder page.
6. README rows, the Voices descriptions in `TOPICS`, CHANGELOG (removing the synthesizer is a
   breaking change for 0.8.0), and `docs/roadmap-done.md`.

The audit's sitting on CheapTrick and D4C rebases after step 1. It isn't affected by step 3, since
the helpers it uses stay in `views/world.py`. Its sitting on WORLD's synthesis is no longer needed.

## Decisions

- **D1. Page order and names** as in "Proposed pages". (Recommended.) The alternative is to keep
  "Two talkers" as the name of page 3 and add a separate voice-quality page. That is one more page,
  and it splits "how voices differ" in two.
- **D2. The recipe lives in the gallery, not in a library function.** (Recommended.) It is a few
  lines on page 9, built from `harmonic_complex` and an STFT mask, and the page shows each step.
  The alternative is a new `aperiodicity=` option on `harmonic_complex` that adds the shaped noise
  itself. Choose it if the recipe turns out to be needed outside the gallery.
- **D3. "How voices differ" measures indicators** (H1–H2, H1–A3, noise share, vocal-tract length
  ratio) on both talkers. (Recommended.) The alternative is to describe them in prose only and
  leave the measuring out.
- **D4. STRAIGHT in prose and math only, no code.** (Recommended.) A sketch of its smoothing could
  be written from the paper's equations as gallery code, like the clean-room texture code. It
  would be sonore's own and would need a checker.
- **D5. `views/world.py` keeps the shared WORLD helpers**, minus the synthesis. (Recommended,
  because nothing that imports them moves.) The alternative is to move them into
  `views/spectral_envelope.py` and delete `views/world.py`.
- **D6. The analysis ports stay exact**, tested against stored pyworld output. (Recommended.) The
  alternative is to let them drift from WORLD when sonore finds something better, under the rule
  that a measure changed from WORLD gets its own name.
- **D7. Noise-band examples he2 and he3 move to the Vocoder page.** (Recommended.) The
  alternative is to keep them on page 9 as one more way to put the envelope back.

## References

- Kawahara, Masuda-Katsuse & de Cheveigné (1999). Restructuring speech representations using a
  pitch-adaptive time-frequency smoothing and an instantaneous-frequency-based F0 extraction.
  *Speech Communication* 27, 187–207.
- Kawahara, Morise, Takahashi, Nisimura, Irino & Banno (2008). TANDEM-STRAIGHT: a temporally
  stable power spectral representation for periodic signals and applications to
  interference-free spectrum, F0, and aperiodicity estimation. *Proc. ICASSP*, 3933–3936.
- Morise (2015). CheapTrick, a spectral envelope estimator for high-quality speech synthesis.
  *Speech Communication* 67, 1–7.
- Morise (2016). D4C, a band-aperiodicity estimator for high-quality speech synthesis. *Speech
  Communication* 84, 57–65.
