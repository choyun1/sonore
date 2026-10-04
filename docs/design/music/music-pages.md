# Music pages: tuning and temperament, and the pipe organ

Cho asked on 2026-10-04 for a musical acoustics section in the gallery: timbre, temperament
(just intonation against equal temperament, and A4 = 432 Hz as one reference pitch among
many), and Lissajous figures. Cho chose temperament first, and added a page on the pipe organ:
stops as a kind of additive synthesizer, where adding stops changes the timbre of one note
rather than adding notes, except in gap registrations. This note proposes both pages and the
little library code they need. It adds no code; the decisions at the end come first.

Status: Cho answered on 2026-10-04. Accepted: D1, D3, D6, D7, D10. Changed: D4 (animated
Lissajous figures, with a short melody and chords in several temperaments), D8 (a paragraph
on historical temperaments pointing to sources, and one cautionary sentence on 432 Hz), D9
(the timbre page comes first, as a placeholder to begin with, and mentions vowels as timbres),
so the pages run timbre, temperament, organ. Taken as the default while unanswered: D2 (after
the FrequencyScale PR, #123) and D5 (a: synthesis now). The checker covers C1–C10.

## How the claims are verified

As in the other design notes, each claim is tagged:

- **[proof]**: a short argument given here.
- **[check]**: a number printed by `tools/check_music_claims.py` (C1–C9). It uses only Python
  and NumPy, shares no code with sonore, and runs in under a second.
- **[source]**: a published work (see References, which says how far each was checked).

Anything else is marked as an estimate, as Cho's experience as an organist, or as inferred.
Neither page makes a perceptual claim of its own: where a page says how something sounds, it
cites a source or says it is the page's description of the example, not a tested result.

## What sonore already has

- `so.harmonic_complex` with any amplitudes and starting phases, and an `amplitudes(t, f, n)`
  function, so every partial can have its own envelope (an attack, a decay). With a fixed F0 it
  also accepts non-integer partial numbers (tried `harmonics=[1, 2.01, 3.03]`), so stretched
  partials and slightly mistuned ranks work today, although the docstring does not say so.
- `so.pure_tone`, sawtooth, square and pulse waves, `so.gaussian_noise` (for wind noise and
  the chiff of a flue pipe), `so.resonator` (for a reed's or a room's resonances),
  `so.amplitude_modulate` (a tremulant's level change), `so.reverb` tools.
- Beats and roughness on the Classic stimuli page (keys b1–b3, a1).
- Spectrum, STFT and reassigned spectrogram, F0 tracking, cepstrum: enough to show a
  registration's spectrum and that its pitch stays put.
- Stereo sounds, so a Lissajous figure is the left channel plotted against the right.

## What is missing

1. Tuning helpers: cents from a ratio and back, and the frequency of a named note for a chosen
   reference A4.
2. A Lissajous plot.
3. Tuning tables (equal, just, Pythagorean, quarter-comma meantone). Page code, not library code
   (D3).
4. Organ pipe spectra. sonore has no pipe models and no organ recordings (D5).

## Page 1: Timbre (placeholder first)

See D9 for its contents; it opens the group and is published first as a short placeholder.

## Page 2: Tuning and temperament

Sections, each with listening examples:

1. **A reference pitch is a choice.** The same short phrase at A4 = 440 Hz and at 432 Hz. The
   432 Hz version is 31.8 cents lower, about a third of a semitone [check C1]: a transposition
   like any other, and the spectrogram on a log frequency axis is the same picture moved down.
   ISO 16 (1975) fixes A4 = 440 Hz [source]; earlier references varied widely from place to
   place and century to century (Haynes, 2002; Ellis, 1885) [source]. The health claims made
   for 432 Hz rest mainly on one pilot study: 33 listeners heard the same film music at each
   tuning in two 20-minute sessions, and the authors reported a lower mean heart rate at
   432 Hz, by 4.79 beats per minute with p = 0.05 (Calamassi & Pomponi, 2019) [source]. The
   page gives this one sentence with a word of caution (D8) and says what the two sounds
   differ by.
2. **Intervals are ratios.** The harmonic series of one note; two notes a just fifth (3:2) or
   major third (5:4) apart share partials exactly, so nothing beats [proof].
3. **The comma.** Twelve just fifths overshoot seven octaves by the Pythagorean comma,
   23.5 cents; four fifths overshoot a just major third plus two octaves by the syntonic comma,
   21.5 cents [check C3]. So no tuning of twelve notes has every fifth and third just [proof].
4. **Equal temperament.** Every semitone 100 cents. The fifth is 2.0 cents narrow and the major
   third 13.7 cents wide of just [check C2]. Over A3 = 220 Hz the nearest shared partials beat
   at 0.74 Hz for the fifth and 8.7 Hz for the major third [check C4]. Examples: a just and an
   equal-tempered major triad, one after the other.
5. **Meantone** (optional): quarter-comma meantone makes the thirds just and the fifths
   696.6 cents, leaving one wolf fifth of 737.6 cents [check C3]. Many historical organs were
   tuned this way (Cho may have more to say than the page should; source to be found).
6. **Lissajous figures.** One note in each channel, drawn as left against right. A just 3:2 pair
   draws a closed figure that stands still. An equal-tempered fifth over A3 passes through every
   shape of the figure once in 1.34 s, the reciprocal of 2 f_upper − 3 f_lower [check C5]; the
   tempered third turns more than ten times faster [check C4, inferred from the beat rate].

## Page 3: The pipe organ

The organ's registration is additive synthesis: each stop is a rank of pipes, one per key,
and drawing stops adds ranks. A stop's footage names its pitch: 8′ sounds the written note,
4′ an octave higher, 16′ an octave lower, and a rank at 8/n feet reinforces harmonic n of the
8′ stop [check C7; source: organ-stop literature]. The footage is nominal: an ideal open pipe
8 ft long sounds about 70 Hz, while low C, C2, is 65.4 Hz and would need 8.6 ft [check C6];
the name fixed a pitch class and stayed.

The analogy Cho drew with an analog synthesizer holds most closely for the tonewheel organ,
whose drawbars copy the organ footages (16′, 5⅓′, 8′, 4′, 2⅔′, 2′, 1⅗′, 1⅓′, 1′) [source:
Wiltshire]. A typical analog synthesizer works the other way, subtractively: a rich oscillator
(a sawtooth) shaped by a filter. The page would show both: a registration built up partial by
partial, and a sawtooth through `so.resonator`, a reed-like tone made by subtraction.

Sections:

1. **One stop.** A principal 8′, a flute 8′, a stopped flute 8′ (mostly odd harmonics, as a
   pipe closed at one end has [source: Fletcher & Rossing]), a reed 8′. Same pitch, different
   spectra.
2. **Building a registration.** 8′, then +4′, +2⅔′, +2′, +1⅗′ (the cornet), then a mixture.
   The pitch track (`so.f0_track`) stays on the same note while the spectrum fills in: timbre
   changes, the pitch does not. Gap registrations (for example 8′ + 2′ with no 4′, or 8′ + 1⅓′)
   are where the separate ranks may stand out as pitches; that is Cho's experience as an
   organist, and the page would present the examples and say so, not claim a perceptual result.
3. **Mutations are tuned pure.** Mutations and the fifths and thirds of mixtures are tuned
   pure to the unison [source: organ-stop literature]. A tierce tuned equal-tempered instead,
   as on a tonewheel organ, would be 13.7 cents sharp of the 5th harmonic and beat with it at
   10.4 Hz on middle C; a tempered twelfth would beat at 0.89 Hz [check C8]. The example plays
   both. This is where the two pages meet.
4. **Celeste and tremulant.** A celeste rank tuned a few cents sharp of its partner beats at
   about 0.45–1.5 Hz on middle C for 3–10 cents [check C9; the detuning range is an estimate
   for illustration, not a survey of organs], the same beats as on the Classic page. A
   tremulant modulates the wind, and with it level and pitch (rate and depth to be sourced).
5. **Attack.** Flue pipes speak with a short noisy transient (chiff); a registration without it
   sounds less like a pipe [description, to be sourced or presented as Cho's judgement].
   The wider point, that the onset carries much of what identifies an instrument, is
   standard: McAdams et al. (2023) left the attack of every tone untouched "as it contributes
   significantly to instrument identification", citing Saldanha & Corso (1964), and in their
   own results listeners never confused a sustained instrument with an impulsive one, which
   they attribute to the temporal envelope and onset [source, read]. The cue is not the onset
   alone: McAdams et al. (1995) report that in Wedin & Goude (1972) removing the attack
   "seemed to have only a slight effect on the perceptual structure" of similarity ratings
   (mean dissimilarities with and without it correlated at .92), and summarize Iverson &
   Krumhansl (1993) as finding the spectral centroid in spaces built from complete tones,
   attacks only, and attacks removed alike [source, read at second hand]. Siedenburg (2019)
   pins down which part of the onset matters [source, read]. Ten instruments (Vienna
   Symphonic Library, C4 to B4), 64 ms excerpts, 18 trained listeners, chance 10%: excerpts from
   the onset were identified 77% of the time, the same with the rapidly varying transient
   removed 71%, and excerpts from the middle of the tone 52%. So the onset matters (moving the
   excerpt cost 25 points) but mostly through the slower buildup of the partials; the
   transient itself (a hammer's knock, a pipe's chiff) cost 6 points. Siedenburg also
   summarizes Saldanha & Corso (1964): about 40% correct overall, 15 points lower with the
   onsets cut. The page gives those numbers, and for the organ draws the point that a chiff
   alone does not make a pipe: how fast each harmonic builds up should be part of the
   synthesis too (an estimate until a source for pipe onsets is found). It links to the timbre
   page (D9).

## Decisions

- **D1. Placement.** A new fourth TOPICS group, "Music", folder `docs/gallery/music/`, with
  `timbre.html` ("Timbre"), `temperament.html` ("Tuning and temperament") and `organ.html`
  ("The pipe organ"), in that order (order per Cho, D9). Accepted. Recommended. Alternative: both pages in Stimuli.
- **D2. Tuning helpers in the library.** `so.ratio_to_cents(ratio)`, `so.cents_to_ratio(cents)`
  and `so.note_to_freq(note, a4=440.0)` (note names like "A4", "C#3", "Bb2", equal temperament),
  in `core/utils.py` beside the ERB and mel converters. Recommended: three short functions the
  gallery and users would reuse. Timing: they land only after the "Source audit and API manual
  plan" PR, which moves the filterbank's frequency scale into `core/utils.py` as
  `so.FrequencyScale` and adds a `cents_scale(reference=440.0)` helper there. These functions
  go next to it, and they share one cents formula with it rather than each keeping its own.
  ("Scale" in this note means a tuning table; the filterbank's is always `FrequencyScale`.)
- **D3. Scales stay page code.** Just, Pythagorean and meantone tables are a few lines each in
  the temperament script. Recommended: a scale class would be a large addition for one page. Accepted.
- **D4. Lissajous figures, animated.** Cho wants them to move with the sound. Two parts:
  `so.plot_lissajous(sound, ax=None, duration=None, start=0.0)` in `plotting.py` for a still
  figure (left channel against right over a short stretch), and a second kind of `live` canvas
  in build.py that draws the figure for the stretch under the playhead as the sound plays
  (today's live canvas draws a colour image, so this is new code in build.py's script and
  `live_json`). The music is a short public-domain tune with chords, Twinkle, Twinkle, Little
  Star, played in equal temperament, 5-limit just intonation on C, Pythagorean and
  quarter-comma meantone. A Lissajous figure takes two signals, so the figure plots the bass
  (left) against the melody (right), and the page says so; the chord's inner notes are heard
  but not drawn. Twinkle uses only the I, IV and V chords, and just intonation fixed on C keeps
  all three exactly just (thirds 386.3 cents, fifths 702.0) [check C10], so the tune shows just
  intonation at its best; a short extra phrase with the ii chord (D minor) shows the cost, a
  fifth of 680.4 cents and a minor third of 294.1 [check C10]. Accepted as changed by Cho.
- **D5. Where the pipe sounds come from.** (a) Synthesis only: each stop a `harmonic_complex`
  with a spectrum per stop family, a chiff from filtered noise, a little wind noise. The
  spectra would come from published measurements where they exist and be labelled estimates
  where they don't. For the principal there are measured numbers: Harrison & Thompson-Allen
  (1998) give the SPL of each harmonic of the Great No. 1 Diapason of the Newberry organ at C2
  (7 harmonics), C4 (11) and C6 (7), measured in the hall 29.7 m away. They report that the
  partials were harmonic within their FFT resolution, that the spectral envelope at C4 to C6
  differs from the envelope at C2, C3 and C7 to C9, and that wind noise was 0 to 15 dB(C)
  above the hall's noise [source]. So the principal would use those three spectra,
  interpolated in dB across the keyboard, rather than one envelope for every key; the room is
  part of those numbers. Flutes and reeds still need a source (candidate: Fletcher, Blackham
  & Christensen, 1963, not yet read). (b) Cho records
  single stops on an instrument Cho plays, one note per stop and a few registrations, which the
  page analyses next to the synthesis. Recommended: (a) for the page now, and (b) if Cho is
  willing, since recordings would replace the estimates with measurements Cho owns.
- **D6. Mutations pure by default.** The organ page tunes mutations pure, and shows tempered
  mutations only as the contrast in section 3. Recommended, following the source. Accepted.
- **D7. Which gap registrations.** 8′ + 2′ and 8′ + 1⅓′ as first examples; Cho to choose or add
  the ones Cho uses. Accepted.
- **D8. History and 432 Hz.** One paragraph on historical temperaments (Pythagorean,
  meantone, well temperaments, equal temperament), pointing readers to sources rather than
  arguing them; Haynes (2002) and a temperament history to be chosen and read before the page
  cites them. Then one sentence on 432 Hz: that it is one more reference pitch, that the
  health claims made for it rest on a small pilot study (Calamassi & Pomponi, 2019), and that
  readers should not take that study as establishing health effects. Accepted as changed by Cho.
- **D9. Timbre page first, built on timbre spaces.** Cho wants timbre as the first page of the
  group, published first as a placeholder (a short introduction and a "to come" note) and
  filled in later. Its backbone would be the McGill/IRCAM timbre-space work: listeners rate how different
  pairs of sounds are, and multidimensional scaling places the sounds in a space. In McAdams
  et al. (1995) (18 synthesized instrument-like tones at E-flat 4, 311 Hz; 88 listeners) the
  three shared dimensions correlated with log attack time (r = −.94), spectral centroid
  (r = −.94) and spectral flux (r = .54) [source, read]. That gives the page a natural shape:
  synthetic tones that move along one dimension at a time (attack time, then brightness, then
  how much the spectrum changes over time), which sonore can make with `harmonic_complex` and
  an `amplitudes(t, f, n)` function. Synthetic tones are fair here because the study itself
  used synthetic tones. The page also connects timbre to speech: vowels differ by their
  spectral envelopes, the formants, at the same pitch and level, so different vowels are in
  effect different timbres, which links to the Formant synthesis and Cepstral analysis pages
  (a framing, not a perceptual claim of the page's own). Siedenburg (2019) suggests one more synthetic demonstration: the
  same tone with all partials starting together, with them building up at different rates,
  and with a transient burst added, to hear which changes more. The onset-removal demonstration (Saldanha & Corso, 1964) still needs
  recordings of real instruments under a licence the gallery can publish; McAdams et al.
  (2023) used the Vienna Symphonic Library and the McGill University Master Samples, neither
  of which is known to allow redistribution (not checked).
- **D10. Timbre descriptors as views (accepted).** With D9: `log_attack_time`, `spectral_centroid` and
  `spectral_flux`, one-way views (philosophy.md "Views may discard information"), defined as
  McAdams et al. (1995) give them after Krimphoff et al. (1994): attack time from 2% of the
  envelope's maximum to the maximum, on a log scale; the mean over the tone of the
  instantaneous centroid in a running 12 ms window; flux as the mean correlation between
  amplitude spectra in adjacent windows. Not needed for the temperament or organ pages; decide
  when D9 starts.

## Patch plan, after the decisions

1. The Music group with the Timbre placeholder page: TITLES and TOPICS entries,
   `docs/gallery/music/timbre.py`, the README gallery list.
2. After #123 merges: D2 helpers beside `cents_scale` in `core/utils.py`, sharing its formula,
   and the D4 still plot, with tests.
3. The animated Lissajous canvas in build.py, then `docs/gallery/music/temperament.py`.
4. `docs/gallery/music/organ.py`.
5. The Timbre page proper, with the D10 descriptors.

Each step is its own PR, and images are built locally by Cho if the cloud matplotlib differs.

## References

- Berger, K. W. (1964). Some factors in the recognition of timbre. *J. Acoust. Soc. Am.* 36,
  1888–1891. Not yet read (title and year only; volume and pages from memory).
- Calamassi, D., & Pomponi, G. P. (2019). Music tuned to 440 Hz versus 432 Hz and the health
  effects: a double-blind cross-over pilot study. *Explore* 15(4), 283–290.
  doi:10.1016/j.explore.2019.04.001. Checked: abstract (design, n = 33, heart-rate result).
  A corrigendum exists (PubMed 32113613), not yet read.
- Ellis, A. J. (1885). On the history of musical pitch, appendix to his translation of
  Helmholtz, *On the Sensations of Tone*. Not yet checked.
- Fletcher, N. H., & Rossing, T. D. (1998). *The Physics of Musical Instruments*, 2nd ed.,
  ch. 17 "Pipe organs", pp. 552–580. Springer. Checked: chapter title and pages only.
- Haynes, B. (2002). *A History of Performing Pitch: The Story of "A"*. Scarecrow Press. Not
  yet checked.
- Harrison, J. M., & Thompson-Allen, N. (1998). Steady-state spectra of diapason class stops
  of the Newberry Memorial organ, Yale University. *J. Acoust. Soc. Am.* 103(1), 626–629.
  doi:10.1121/1.421134. Checked: read in full (PDF from Cho); Tables I–III SPL columns.
- Fletcher, H., Blackham, E. D., & Christensen, D. A. (1963). Quality of organ tones. *J.
  Acoust. Soc. Am.* 35, 314–325. Cited by Harrison & Thompson-Allen; not yet read.
- Iverson, P., & Krumhansl, C. L. (1993). Isolating the dynamic attributes of musical timbre.
  *J. Acoust. Soc. Am.* 94(5), 2595–2603. Not read; findings as summarized by McAdams et al.
  (1995).
- ISO 16:1975. Acoustics — Standard tuning frequency (standard musical pitch). Checked: title
  and the 440 Hz figure from the ISO catalogue entry.
- Organ stop (Wikipedia, read 2026-10-04): footages, mutation table, "mutations are always
  tuned pure", mixture breaks. A secondary source; to be replaced by Fletcher & Rossing or an
  organ-building text before the page cites it.
- McAdams, S., Winsberg, S., Donnadieu, S., De Soete, G., & Krimphoff, J. (1995). Perceptual
  scaling of synthesized musical timbres: common dimensions, specificities, and latent subject
  classes. *Psychol. Res.* 58, 177–192. Checked: read (PDF from Cho); Table 7 correlations, the
  review of earlier MDS work, descriptor definitions. One slip to avoid copying: it attributes
  the clarinet's odd harmonics to a conical bore; the clarinet's bore is cylindrical.
- McAdams, S., Thoret, E., Wang, G., & Montrey, M. (2023). Timbral cues for learning to
  generalize musical instrument identity across pitch register. *J. Acoust. Soc. Am.* 153(2),
  797–811. doi:10.1121/10.0017100. Checked: read (PDF from Cho); stimuli, the attack sentence,
  the sustained/impulsive confusions. It also reports that identification is predicted
  moderately well by spectrograms or modulation spectra.
- Plomp, R., & Levelt, W. J. M. (1965). Tonal consonance and critical bandwidth. *J. Acoust.
  Soc. Am.* 38(4), 548–560. Already cited on the Classic stimuli page.
- Siedenburg, K. (2019). Specifying the perceptual relevance of onset transients for musical
  instrument identification. *J. Acoust. Soc. Am.* 145(2), 1078–1087.
  doi:10.1121/1.5091778. Checked: read (PDF from Cho); experiment 1 method and scores, the
  review of Saldanha & Corso, Clark et al., Elliott and Iverson & Krumhansl. Its
  transient/stationary separation (Siedenburg & Doclo, 2017: two STFT window lengths, 46 ms
  and 3 ms, with grouped shrinkage) is close to sonore's frames and could be a later demo.
- Saldanha, E. L., & Corso, J. F. (1964). Timbre cues and the identification of musical
  instruments. *J. Acoust. Soc. Am.* 36(11), 2021–2026. Not yet read; citation confirmed by
  McAdams et al. (2023).
- Wiltshire, T. Technical aspects of the Hammond organ. electricdruid.net, read 2026-10-04:
  nine drawbars, 91 tonewheels, tempered rather than pure ratios. Not a peer-reviewed source.
