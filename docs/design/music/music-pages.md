# Music pages: tuning and temperament, and the pipe organ

Cho asked on 2026-10-04 for a musical acoustics section in the gallery: timbre, temperament
(just intonation against equal temperament, and A4 = 432 Hz as one reference pitch among
many), and Lissajous figures. Cho chose temperament first, and added a page on the pipe organ:
stops as a kind of additive synthesizer, where adding stops changes the timbre of one note
rather than adding notes, except in gap registrations. This note proposes both pages and the
little library code they need. It adds no code; the decisions at the end come first.

Status: proposal, waiting for Cho's answers to D1–D9.

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
3. Scale tables (equal, just, Pythagorean, quarter-comma meantone). Page code, not library code
   (D3).
4. Organ pipe spectra. sonore has no pipe models and no organ recordings (D5).

## Page 1: Tuning and temperament

Sections, each with listening examples:

1. **A reference pitch is a choice.** The same short phrase at A4 = 440 Hz and at 432 Hz. The
   432 Hz version is 31.8 cents lower, about a third of a semitone [check C1]: a transposition
   like any other, and the spectrogram on a log frequency axis is the same picture moved down.
   ISO 16 (1975) fixes A4 = 440 Hz [source]; earlier references varied widely from place to
   place and century to century (Haynes, 2002; Ellis, 1885) [source]. The health claims made
   for 432 Hz rest mainly on one pilot study: 33 listeners heard the same film music at each
   tuning in two 20-minute sessions, and the authors reported a lower mean heart rate at
   432 Hz, by 4.79 beats per minute with p = 0.05 (Calamassi & Pomponi, 2019) [source]. The
   page reports that study as it is and says what the two sounds differ by. It does not argue
   further.
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

## Page 2: The pipe organ

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

## Decisions

- **D1. Placement.** A new fourth TOPICS group, "Music", folder `docs/gallery/music/`, with
  `temperament.html` ("Tuning and temperament") and `organ.html` ("The pipe organ"), in that
  order. Recommended. Alternative: both pages in Stimuli.
- **D2. Tuning helpers in the library.** `so.ratio_to_cents(ratio)`, `so.cents_to_ratio(cents)`
  and `so.note_to_freq(note, a4=440.0)` (note names like "A4", "C#3", "Bb2", equal temperament),
  in `core/utils.py` beside the ERB and mel converters. Recommended: three short functions the
  gallery and users would reuse.
- **D3. Scales stay page code.** Just, Pythagorean and meantone tables are a few lines each in
  the temperament script. Recommended: a scale class would be a large addition for one page.
- **D4. Lissajous plot.** `so.plot_lissajous(sound, ax=None, duration=None, start=0.0)`
  in `plotting.py`: left channel against right over a short stretch. A turning figure is shown
  as a row of snapshots a fraction of a second apart. Recommended. An animated figure that
  follows the playhead would need a new kind of `live` canvas in build.py (today's draws a
  colour image); possible later.
- **D5. Where the pipe sounds come from.** (a) Synthesis only: each stop a `harmonic_complex`
  with a spectrum per stop family, a chiff from filtered noise, a little wind noise. The
  spectra would be estimates read from published pipe spectra and labelled so. (b) Cho records
  single stops on an instrument Cho plays, one note per stop and a few registrations, which the
  page analyses next to the synthesis. Recommended: (a) for the page now, and (b) if Cho is
  willing, since recordings would replace the estimates with measurements Cho owns.
- **D6. Mutations pure by default.** The organ page tunes mutations pure, and shows tempered
  mutations only as the contrast in section 3. Recommended, following the source.
- **D7. Which gap registrations.** 8′ + 2′ and 8′ + 1⅓′ as first examples; Cho to choose or add
  the ones Cho uses.
- **D8. The 432 Hz paragraph.** Report the Calamassi & Pomponi study's design and result in one
  sentence, state the 31.8-cent difference, and leave it there. Recommended, in line with
  keeping render and perception claims modest.
- **D9. Timbre page later.** A separate timbre page (spectral envelope, attack, brightness) is
  left for after these two; the organ page covers timbre from stops.

## Patch plan, after the decisions

1. D2 helpers and D4 plot with tests; one commit.
2. `docs/gallery/music/temperament.py`, TITLES and TOPICS entries, README gallery list.
3. `docs/gallery/music/organ.py`.

Each step is its own PR, and images are built locally by Cho if the cloud matplotlib differs.

## References

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
- ISO 16:1975. Acoustics — Standard tuning frequency (standard musical pitch). Checked: title
  and the 440 Hz figure from the ISO catalogue entry.
- Organ stop (Wikipedia, read 2026-10-04): footages, mutation table, "mutations are always
  tuned pure", mixture breaks. A secondary source; to be replaced by Fletcher & Rossing or an
  organ-building text before the page cites it.
- Plomp, R., & Levelt, W. J. M. (1965). Tonal consonance and critical bandwidth. *J. Acoust.
  Soc. Am.* 38(4), 548–560. Already cited on the Classic stimuli page.
- Wiltshire, T. Technical aspects of the Hammond organ. electricdruid.net, read 2026-10-04:
  nine drawbars, 91 tonewheels, tempered rather than pure ratios. Not a peer-reviewed source.
