# The Voices group as one arc: from what a voice is to a richer source-filter model

Status: proposed, not built. Cho asked for this on 2026-10-18 and wants to see the design before
anything is implemented. This is the second draft. The first proposed nine pages, and Cho replaced
it with the arc below. The decisions at the end need Cho's answer.

## The arc

Four pages, each one building on the last:

1. **What is a voice.** A general introduction, light on examples. It covers what produces a voice
   and the many ways voices differ, including the ones measured clinically.
2. **The source-filter model.** One page that implements the model and takes it apart. It
   synthesizes a voice by rule, measures each part back (the source's pitch, the filter as
   resonances, the filter as a smooth envelope), rebuilds a voice simply from those parts, and
   ends on where that falls short.
3. **A richer source-filter model.** Each shortfall gets the fix that higher-quality voice
   synthesis adds: an envelope measured to the pitch period (STRAIGHT, CheapTrick), a source that
   is partly noise (aperiodicity), and a resynthesis that uses both.
4. **Rebuilding and changing a voice.** The richer model at work: pitch, formants, range, and
   moving toward another talker.

Cho's reasons, from the thread: pitch tracking, finding formants, the spectral envelope and formant
synthesis are all about implementing the source-filter model, so they belong on one page.
Higher-quality synthesis is that same model with more put into it. How voices differ belongs with
the introduction, as the list of what a model could capture.

The rules from the gallery rewrite (`gallery-rewrite.md`) still hold. Each topic has one owner
page, and the two talkers are paired rather than the female talker coming last. An example that
moves keeps its key, so old links still land on it (the `moved` redirect in
`docs/gallery/build.py`). Pages that go away become "Moved" stubs (`write_moved_stubs`).

## What is there now

Seven pages in the Voices group. Their gallery sources come to about 3,700 lines: Two talkers
(326), Formant synthesis (712), Finding formants (430), Pitch tracking (448), Spectral envelope
(590), Source and aperiodicity (565) and Rebuilding and changing a voice (972). Several pages
explain the source and the filter in passing, each in its own words (fv1 to fv4, fs1, tk5, c3 to
c5, ap1 to ap3, he1). WORLD's synthesis is used on two pages, ap10 and vc2 to vc19.

## 1. What is a voice (new; replaces Two talkers)

Mostly prose, with a few figures and sounds. It sets up the vocabulary the other three pages use.

**What makes a voice.** Air from the lungs, the vocal folds as a source (a buzz at the fundamental
frequency, or turbulence that hisses), and the vocal tract from larynx to lips as a filter that
shapes it. This is a paragraph and one diagram. The model itself is page 2's.

**How voices differ**, each item stated as what it is in the source or the filter, and how it is
measured:

- **Pitch:** F0 level, range and variability, as mean, SD and range of an F0 track.
- **Cycle-to-cycle irregularity:** jitter (period perturbation) and shimmer (amplitude
  perturbation), as clinical voice reports give them.
- **Noise in the voice:** breathiness and hoarseness, measured as the harmonics-to-noise ratio and
  cepstral peak prominence (CPP, the height of the cepstrum's pitch peak above its trend).
- **Phonation type:** pressed (strained), modal and breathy voice, read from the source spectrum's
  tilt, H1–H2 and H1–A3. In the LF model this is Rd.
- **Vocal effort:** louder voices have a flatter spectrum and higher F0, not only more level.
- **Size:** vocal-tract length, which scales every formant. It is a large part of why the male and
  female talkers differ, and of why children sound different again.

Each item is defined in a sentence or a formula. It's linked to the page that implements it, or
marked as not in sonore (jitter and shimmer need period-by-period marks, which sonore doesn't
compute). Clinical rating scales such as GRBAS and CAPE-V are named as how voices are judged by
ear, with no listening claims made.

**Examples**, few on purpose:

- the sentence read by both talkers (tk1)
- their pitch tracks and average envelopes (tk2, tk3)
- one vowel in three voices, pressed to breathy (fq3, moved from Formant synthesis)
- a small table of the indicators sonore can already compute, measured on both talkers: F0 level
  and range, H1–H2, CPP, the noise share from `so.band_aperiodicity` in a few bands, and a
  formant-scaling ratio from `so.formant_track`

None of the table's numbers are measured yet. They are computed when the page is built, and the
page says what each one leaves out. The long-term spectra (tk4) and "harmonics sample the
envelope" (tk5) move to page 2.

## 2. The source-filter model (merges four pages)

This merges Formant synthesis, Pitch tracking, Finding formants and Spectral envelope, about 2,180
lines of gallery source, so it has to be cut down, not stacked. The rule for cutting: one or two
examples per idea, both talkers where the talker matters, and no example that repeats another
page's point.

**a. Source times filter.** The model in prose and math: `s = e * h`, `S(f) = E(f) H(f)`.
Harmonics at `k F0` sample `|H|`, and noise shows all of it. Keep fv1 to fv4 (a vowel built piece
by piece), fs1 (source plus filter) and tk5 (harmonics sample the envelope).

**b. Synthesis by rule.** Keep fv5 (the same vowel from `so.klatt_synthesize`), fw1 and fw2 (six
vowels and their chart), fc1 to fc3 (ba, da, ga), fn1 and fn2 (sa, ha), fq1 and fq2 (the LF pulse
and its spectra at three Rd values), and fl1 (sonore says its name). Cut fw3 (the seven-step
continuum), fp1 (cascade and parallel, a detail of Klatt's design, left to its docstring), fq4
(the 1980 source against LF) and fq5 (relaxing into breathy voice; page 1's fq3 covers it).

**c. Measuring the source: pitch.** Keep c2 (the cepstrogram and its pitch), f1 (cepstral F0, a
tracker, and Harvest's stored track), pt1 (octave and voicing errors), and hp0 with hp2 (the
sentence, and a sawtooth on `so.f0_track`). Cut hp1 and hp3, the sawtooths on the other two tracks.
pt1 already shows where they differ.

**d. Measuring the filter as resonances: formants.** Keep ft1 (one vowel, four envelopes, the
formants LPC finds), ft4 (formant tracks of both talkers) and ft5 with ft6 (copy synthesis of
each talker, which closes the loop with section b). Cut ft2 (given against recovered formants) and
ft3 (the estimate as the pitch rises). Section f carries the high-F0 problem instead.

**e. Measuring the filter as a smooth curve: the envelope.** Keep c1 (one time window, its
cepstrum and lifter), c3 to c5 (the voice split into envelope and harmonics) and m1 with m2 (MFCCs:
one vowel at three pitches, the sentence as MFCCs). Cut m3 (two spectra, one set of MFCCs). Move
the lifter-against-CheapTrick comparison in c1 to page 3, so c1 shows the lifter alone.

**f. Putting it back, simply, and where it falls short.** Keep he1 (harmonics shaped by the
envelope), hu1 and hu2 (silence or shaped noise in the unvoiced gaps) and tk4 (both talkers'
long-term spectra, as the check that the copy has the right overall spectrum). The page ends on
three shortfalls, each pointing to page 3:

- the lifter's envelope ripples or loses peaks at high F0, as on the female talker
- voicing is all or nothing, so a breathy vowel is either clean harmonics or plain noise
- the phases are whatever we choose

## 3. A richer source-filter model (new; absorbs Source and aperiodicity)

The page Cho first asked for ("higher quality voice synthesizer"), in prose and math, ordered by
the shortfalls page 2 ends on.

Cho (2026-10-18): the math and the technique are explained plainly, because the original papers
and the old STRAIGHT code were hard to follow (many unexplained variable names). Each method is
built up step by step: what goes wrong without the step, the step in one sentence, then its
equation with every symbol named in words, and a figure of what the step does to one real time
window. The page uses the names in sonore's code, not the papers' symbols, except where an
equation needs a symbol, and then the symbol is defined next to it.

**a. An envelope measured to the pitch period.**

- **The problem.** A periodic signal under a fixed window leaves a ripple at F0, either in
  frequency (a short window) or in time (a long one).
- **STRAIGHT** (Kawahara, Masuda-Katsuse & de Cheveigné, 1999): a window sized to the pitch period,
  smoothing in frequency over F0 with a kernel that keeps the harmonic peaks' level, and a
  complementary window that fills the dips in time.
- **TANDEM-STRAIGHT** (Kawahara et al., 2008): two power spectra half a period apart, added, so the
  fluctuation over time cancels.
- **CheapTrick** (Morise, 2015), as `so.pitch_adaptive_envelope`: a Hann window three periods
  long, whose power is nearly constant over time, then a moving average over `2 F0 / 3`, then
  liftering by `sinc(F0 q)` times the recovery lifter `(1 - 2 q1) + 2 q1 cos(2 pi F0 q)`. Each step
  gets its formula and a figure.
- **Examples:** the lifter against CheapTrick on one time window (moved from c1), a window-length
  sweep showing the ripple trade between frequency and time, both talkers' envelopes (the higher
  voice shows the difference most), and the effect of `q1`.

STRAIGHT is prose and equations only. sonore has no STRAIGHT code and this plan writes none (D4).

**b. A source that is partly noise.** Keep ap1 to ap3 (buzz, hiss, both), ap4 to ap6 (fit the
harmonics, keep the rest), ap7 (renamed "What D4C reports") and ap8 (the sentence's envelope and
aperiodicities).

**c. Phase.** Keep hf0 to hf3 (cosine, Schroeder and random phase on a moving pitch), moved from
the voice-change page, since the phases are one of the shortfalls.

**d. Putting it back.** The recipe that replaces WORLD's synthesis (below), built step by step on
the page. Keep ap9 to ap13 (the sentence, the recipe with D4C and with the harmonic residual, no
noise, nothing but noise). Their sound changes, because the recipe replaces WORLD's synthesis. One
new example compares the page-2 resynthesis with this one on both talkers.

## 4. Rebuilding and changing a voice (condensed)

Keep vc1 (the sentence) and vc2, now the recipe's resynthesis, under "Resynthesis" (no longer
"Resynthesis with WORLD"). Keep the pitch and formant changes (vc3 to vc6), the phase vocoder
(vc7: the sentence a fifth up, pitch and formants moving together, as the contrast to the
source-filter route, with a link back to the Phase vocoder page, which stays in "Seeing and
changing sound" because it works on any sound), pitch range (vc8 to vc10, hm3), the aperiodicity kept and warped (vc11, vc12), toward
another talker (vc14, vc15, vc19), and any pitch track, any envelope (vc17, vc18), all synthesized
by the recipe. hm4 (whispered) stays. he1, hu1, hu2 and hf0 to hf3 move to pages 2 and 3. he2 and
he3 (noise bands, the channel-vocoder idea) move to the Vocoder page, which owns that idea.

## The recipe that replaces WORLD's synthesis

For each time and frequency, with envelope `S`, noise share `a` (`Aperiodicity.share`) and F0:

- **Harmonics:** `so.harmonic_complex(duration, fs, f0, amplitudes=...)`, with harmonic `n`'s power
  equal to `S (1 - a)` over one harmonic spacing. A power spectral density `P` with a harmonic
  every `F0` gives `a_n^2 / 2 = 2 F0 P / fs`.
- **Noise:** white noise, its STFT magnitude multiplied by `sqrt(S a)` (by `sqrt(S)` where
  unvoiced), inverted. The gallery already does this for the unvoiced gaps (hu2).
- **Output:** the two added together.

What it loses against the port is the minimum-phase pulse shape and agreement with WORLD sample
for sample. Cho has said the second doesn't matter.

**Measured:** with the same envelope and no noise, `harmonic_complex` and the port come within
0.11 dB of each other on the steady test vowels (`views/voice-change.md`, C8).

**Not measured:** the noise half, its level calibration, and the score on the recordings. A
checker, `tools/check_voice_recipe_claims.py`, reports these before any page is rebuilt: the
recovered noise share against the truth on the synthetic vowels, and the level-free log mel
distance on both recordings, as in C11. If the noise half measures badly, the plan comes back to
Cho.

## Writing CheapTrick and D4C from the papers (accepted)

Cho asked on 2026-10-18 whether BSD-3 can be avoided. It can. Copyright covers WORLD's code, not
the methods, and the current ports are under BSD-3 only because they were written from WORLD's
C++. Rewriting them from the CheapTrick (2015) and D4C (2016) papers alone, without reading
WORLD's source, makes them sonore's own, as the texture code is. `LICENSE-WORLD` and the file
headers then go. The cost is exactness. Choices that are only in WORLD's code (q1 = −0.15 where the
paper gives −0.09, WORLD's noise generator, MATLAB-style rounding, the FFT sizes) would follow the
paper instead, and the distance to pyworld, kept as a development-time comparison, would be
measured and reported. Before any code, each paper is checked against what the port does, and
anything the papers leave out is listed for Cho (D8).

## What leaves the library

`so.source_filter_synthesize` and its tests go, along with the synthesized arrays in
`tests/data/world_reference.npz`, the synthesis parts of `tools/crosscheck_world_vocoder.py` and
`tools/check_world_claims.py`, and the two DIFFERENCES_FROM_WORLD items about synthesis. The
`back_to_sound` lines of the envelope, aperiodicity, F0, LPC, cepstrum and MFCC views point to
`so.harmonic_complex` and page 3's recipe instead. `views/world.md` and `views/voice-change.md` get
a dated note.

The rest of `views/world.py` stays as the home of the helpers the analysis ports share: WORLD's
noise generator, its FFT sizes, MATLAB rounding, the reading of an F0 track onto time windows, and
DIFFERENCES_FROM_WORLD (D5). If the analyses are rewritten from the papers (D8), what they still
need moves next to them, and `views/world.py` goes.

## Order

Cho's order (2026-10-18): the rewrite from the papers comes before any gallery change.

1. PR #238 (renames and WORLD's license), merged 2026-10-18.
2. **The rewrite of CheapTrick and D4C from the papers** (D8, accepted by Cho's ordering). It is
   done clean-room:
   - **The gap list.** One session compares each paper with what the port does, and lists every
     choice the paper leaves open as a question for Cho, not as code.
   - **The writing.** A fresh worker writes the new functions, given only the papers, the
     answered questions and the public signatures. It does not open the port's code, its checker,
     or WORLD's source.
   - **The comparison.** Each function is then measured against pyworld, as a development-time
     comparison, and the distance is reported. When the rewrite replaces the ports,
     `LICENSE-WORLD` and the file headers go.

   The papers have to be uploaded first (CheapTrick 2015, D4C 2016, and the WORLD overview 2016).
3. The recipe and its checker, with the numbers added here, on the rewritten analyses.
4. Remove `so.source_filter_synthesize`, and move the pages that used it to the recipe.
5. Page 2 (the merge). It is the largest step and one PR.
6. Page 3, then page 1, then page 4's trim and the move to the Vocoder page.
7. README rows, the Voices entries in `TOPICS`, the CHANGELOG (removing the synthesizer and
   replacing the ports are breaking changes for 0.8.0), and `docs/roadmap-done.md`.

The audit's sitting on WORLD's synthesis is no longer needed. Its CheapTrick and D4C sitting
reviews the rewrite instead of the port.

## Decisions

- **D1. Four pages in the order of "The arc".** (Recommended.)
- **D2. The recipe lives in the gallery, not in a library function.** (Recommended.) The
  alternative is an `aperiodicity=` option on `harmonic_complex`, if the recipe is needed outside
  the gallery.
- **D3. Page 1 measures only what sonore already computes**, in one small table: F0, H1–H2, CPP,
  noise share, formant scaling. Jitter, shimmer and HNR are defined but not computed.
  (Recommended.) The alternative is adding period marking for jitter and shimmer, which is new
  library work.
- **D4. STRAIGHT in prose and math only, no code.** (Recommended.)
- **D5. `views/world.py` keeps the shared helpers while the analyses are ports.** (Recommended.)
- **D6. Exactness of the analyses follows D8.** They stay exact if they stay ports. If rewritten,
  they report their distance to WORLD.
- **D7. The noise-band examples he2 and he3 move to the Vocoder page.** (Recommended.)
- **D8. Rewrite CheapTrick and D4C from the papers**, retiring `LICENSE-WORLD`. Accepted by Cho
  (2026-10-18), to be done before the gallery changes.
- **D9. The cuts on page 2** as listed in its sections: fw3, fp1, fq4, fq5, hp1, hp3, ft2, ft3
  and m3. (Recommended.) Any of them can be kept instead.

## References

- Hillenbrand, Cleveland & Erickson (1994). Acoustic correlates of breathy vocal quality.
  *Journal of Speech and Hearing Research* 37, 769–778. (CPP.)
- Kawahara, Masuda-Katsuse & de Cheveigné (1999). Restructuring speech representations using a
  pitch-adaptive time-frequency smoothing and an instantaneous-frequency-based F0 extraction.
  *Speech Communication* 27, 187–207.
- Kawahara, Morise, Takahashi, Nisimura, Irino & Banno (2008). TANDEM-STRAIGHT: a temporally
  stable power spectral representation for periodic signals and applications to
  interference-free spectrum, F0, and aperiodicity estimation. *Proc. ICASSP*, 3933–3936.
- Kempster, Gerratt, Verdolini Abbott, Barkmeier-Kraemer & Hillman (2009). Consensus Auditory-
  Perceptual Evaluation of Voice: development of a standardized clinical protocol. *American
  Journal of Speech-Language Pathology* 18, 124–132. (CAPE-V.)
- Hirano (1981). *Clinical Examination of Voice*. Springer. (GRBAS.)
- Morise (2015). CheapTrick, a spectral envelope estimator for high-quality speech synthesis.
  *Speech Communication* 67, 1–7.
- Morise (2016). D4C, a band-aperiodicity estimator for high-quality speech synthesis. *Speech
  Communication* 84, 57–65.

The citations for page 1's clinical measures are checked against the publications when that page
is written.
