# Classic auditory demonstrations, rebuilt with sonore

Cho agreed on 2026-10-29 to gallery pages for the two best-known sets of auditory demonstrations:
the ASA *Auditory Demonstrations* CD (Houtsma, Rossing & Wagenaars, 1987; 39 demonstrations) and
Bregman & Ahad's *Demonstrations of Auditory Scene Analysis* (1996; 41 demonstrations). Many
hearing scientists learned from these two discs, so pages that rebuild them give a later
announcement something familiar to point to, and the stimuli are ones sonore-infer can use.

This note lists every demonstration, says what sonore already makes, and proposes pages. It adds
no code; the decisions at the end come first.

Status: draft for Cho, 2026-10-29.

## What the pages are, and what they are not

- **The stimuli, rebuilt from the published parameters.** Both booklets give frequencies, levels,
  durations and timing for nearly every demonstration. Each example on a page is a few sonore
  calls that rebuild the stimulus from those numbers, with the code shown beside the sound, as on
  every other page. The original audio and the booklets' text and figures are not copied: both
  are under copyright (Bregman's booklet says so on its first page). A parameter is a fact and is
  cited; the prose is our own.
- **Not psychophysics.** No page measures a listener, keeps a score, or models what is heard.
  Where a demonstration was a listening test (count the steps, which tone is higher), the page
  describes what the original asked listeners to do and cites it, and leaves it there.
- **Modest about what is heard.** A page says what the booklet or the original paper reports
  listeners hear, attributed ("Bregman & Ahad report that…"), never in sonore's own voice. A
  rebuilt stimulus is described by what it is ("tones of 1400 and 500 Hz, 100 ms each"), not as
  equal to the original.
- **Each demonstration keeps its source number.** An example is titled for its source, e.g. "A
  galloping rhythm that splits (Bregman & Ahad, Demonstration 3)", so readers who know the discs
  can find it.

Other rebuilds exist: Bregman's lab hosts the 1996 audio and booklet, the 1987 CD's audio is on
several university sites, and Slaney's correlogram pages animate many of the ASA tracks. What
these pages add is the stimulus as code, short enough to read, change and rerun.

## Inventory

Each row says how the demonstration would be rebuilt:

- **on page X**: an existing gallery example already shows it. The new page links to it rather
  than repeating it.
- **build**: plain sonore calls (`pure_tone`, `harmonic_complex` with per-harmonic amplitude
  functions or F0 contours, `gaussian_noise`, `bandpass`, `concat`, `mix`, `apply_itd_ild`,
  `klatt_synthesize`, `formant_track`). No new library code.
- **stand-in**: the original used a recording; a synthetic or in-repo substitute keeps the point
  (D5).
- **leave out**: with the reason.

### Houtsma, Rossing & Wagenaars (1987)

| # | Demonstration | How |
|---|---|---|
| 1 | Cancelled harmonics (20 harmonics of 200 Hz, one switched off and on) | build |
| 2 | Critical bands by masking (2 kHz tone in noise of shrinking bandwidth) | build (masked staircase, D3) |
| 3 | Critical bands by loudness comparison (noise bands at constant power) | build |
| 4 | The decibel scale (noise in 6, 3, 1 dB steps; a voice at 25 to 200 cm) | build; the voice is a talker sentence with `so.distance_gain_db` (stand-in) |
| 5 | Filtered noise (low-, high-, band-pass; white against pink) | build |
| 6 | Frequency response of the ear (absolute-threshold staircases) | leave out: depends on the reader's playback level (D3) |
| 7 | Loudness scaling (magnitude estimation) | leave out: a procedure, not a stimulus |
| 8 | Temporal integration (noise bursts of 1 to 1000 ms in a masker) | build (masked staircase) |
| 9 | Asymmetry of masking (1200 and 2000 Hz pulsed tones) | build (masked staircase) |
| 10 | Backward and forward masking (10 ms tone, 250 ms noise, gaps 100, 20, 0 ms) | build (masked staircase) |
| 11 | Pulsation threshold (2 kHz tone alternating with narrowband noise) | build; shares a section with Bregman 28 |
| 12 | Dependence of pitch on intensity (pairs 30 dB apart) | build, low priority: small effect that the booklet says varies between listeners |
| 13 | Pitch salience and tone duration (1 to 128 periods) | build |
| 14 | Influence of masking noise on pitch | build, low priority (small effect) |
| 15 | Octave matching (500 Hz against 985 to 1035 Hz) | build |
| 16 | Stretched and compressed scales (melody against bass) | build with a public-domain melody (D5) |
| 17 | Frequency difference limen (answer-keyed pairs) | leave out: a listening test |
| 18 | Logarithmic and linear frequency scales | build; fits the Tuning page |
| 19 | Pitch streaming (van Noorden's ABA gallop) | build; shares a section with Bregman 3 |
| 20 | Virtual pitch (10 harmonics of 200 Hz, lowest removed in turn) | build |
| 21 | Shift of virtual pitch (800, 1000, 1200 Hz shifted in 20 Hz steps) | build; the Phase vocoder page's shifted partials (keys 14, 14b) do this to a tone, and are linked |
| 22 | Masking spectral and virtual pitch (Westminster chime, low- and high-pass noise) | build |
| 23 | Virtual pitch with random harmonics | build |
| 24 | Strike note of a chime (recorded chime) | leave out until partial data for a chime is found (D5) |
| 25 | Analytic against synthetic pitch (800+1000 Hz, then 750+1000 Hz) | build |
| 26 | Scales with repetition pitch (pulse pairs, Poisson pulse pairs, echoed noise) | build; echoed noise is on the IRN page and is linked |
| 27 | Circularity in pitch (Shepard scale, Risset glide) | on Chroma (keys pc3, pc5) |
| 28 | Effect of spectrum on timbre (bell and guitar built up partial by partial) | stand-in: a synthetic bell from the booklet's partial frequencies, decay times estimated |
| 29 | Effect of tone envelope on timbre (piano chorale, notes reversed) | stand-in, needs a public-domain chorale (D5); the Timbre page's attacks (tb1) are related |
| 30 | Change in timbre with transposition (bassoon) | on Rebuilding and changing a voice (pitch alone against pitch and formants, vc3 and vc5), linked |
| 31 | Tones and tuning with stretched partials (chorale, octave 2.1) | build, needs the same chorale (D5); fits the Tuning page |
| 32 | Primary and secondary beats (1000+1004 Hz; mistuned octave, fifth, fourth) | primary beats on Tones and noises (b1); secondary beats build |
| 33 | Distortion (compressor, half-wave rectifier, square law) | build |
| 34 | Aural combination tones (1000 Hz with 1200 to 1600 Hz) | build; the page's spectrum shows the difference tone is not in the signal |
| 35 | Effect of echoes (hammer and speech reversed in three rooms) | stand-in: a talker sentence in synthetic rooms, reversed; the Rooms page's reversed decay (key 18) is related |
| 36 | Binaural beats (250 and 251 Hz) | on Binaural (key b4) |
| 37 | Binaural lateralization (phase at 500 and 2000 Hz, click ITD, ILD sweep) | build; Binaural's "Timing alone" is linked |
| 38 | Masking level differences (500 Hz, signal phase inverted at one ear) | build |
| 39 | An auditory illusion (Deutsch's octave illusion, 400 and 800 Hz) | build |

### Bregman & Ahad (1996)

| # | Demonstration | How |
|---|---|---|
| 1 | Stream segregation in a cycle of six tones | build |
| 2 | Pattern recognition within and across streams | build |
| 3 | Loss of rhythm by streaming (accelerating gallop, 18 and 1 semitones) | build |
| 4 | Cumulative effects of repetition (2 to 32 gallops) | build |
| 5 | A melody among interleaved distractors | build; the hidden tune is public domain |
| 6 | High and low notes in a Telemann sonata (recording) | leave out: a recording (D5) |
| 7–9 | Streaming in amadinda music: by pitch range, by timbre | stand-in: an interlocking pentatonic pattern written for the page, on a synthetic struck tone (D5) |
| 10 | Streaming by spectral peak position (300 Hz tones, peaks at 300 and 2000 Hz) | build |
| 11 | Streaming of vowels and diphthongs | build with `klatt_synthesize` |
| 12 | Effects of connectedness (glides between alternating tones) | build |
| 13 | Streaming and judgments of timing | build |
| 14 | Streaming of high and low noise bands | build (sharp-edged bands as sums of random-phase tones) |
| 15 | Competition of frequency separations | build |
| 16 | Release of a two-tone target by capturing | build |
| 17 | Crossing trajectories that bounce | build |
| 18 | Isolation of a mistuned harmonic | build |
| 19–20 | Fusion by common frequency change | build (two `harmonic_complex` calls with different F0 contours) |
| 21 | Rate of onset and segregation | build |
| 22 | Rhythmic masking release | build; with comodulation masking release beside it (D8) |
| 23 | Sine-wave speech (Remez's parameters) | stand-in: from the male talker's sentence with `so.formant_track` |
| 24 | Frequency micro-modulation in voice perception (Chowning) | build (vibrato plus low-pass noise as the F0 contour) |
| 25–27 | Capturing a component out of a mixture; competition of groupings | build |
| 28 | Apparent continuity (1000 Hz tone, noise in the gap) | build |
| 29–30 | A glide continuing through noise; no extrapolation of the peak | build |
| 31 | The picket-fence effect with speech | build on a talker sentence |
| 32 | Homophonic continuity and rise time | build |
| 33 | A high residual made by capturing harmonics | build |
| 34 | Capturing a band of noise from a wider band | build |
| 35 | Sequences of narrow-band noises | build |
| 36 | Capturing a glide out of a mixture of glides | build |
| 37 | Changing a vowel by capturing its third harmonic | build (additive, 125 Hz, peaks from the booklet) |
| 38 | Streaming by spatial location (noise bursts panned apart) | build |
| 39 | Spatial streaming and timing | build |
| 40 | Fusion of left and right noise by their correlation | build; Binaural's "Correlation that changes" is linked |
| 41 | Location difference in amadinda music | stand-in (as 7–9) |

Counts: of the 80, 64 are built directly, 8 need a stand-in, 3 are already on a page, and 5 are
left out (ASA 6, 7, 17 and 24, and Bregman 6).

### What was tried

To check the riskiest rows before proposing them, three stimuli were built in scratch code (not
committed):

- **Sine-wave speech** from the male talker's sentence: `so.formant_track`, then three
  `so.harmonic_complex(..., harmonics=[1])` calls whose F0 contours are F1 to F3 and whose
  amplitude functions follow the sentence's short-time RMS at 0, -6 and -12 dB. It runs in about
  1.5 s, and its spectrogram follows the formants through the voiced stretches; the tracks jump
  in the unvoiced stretches, so the page would mask them with the F0 track's voicing. Nobody has
  listened to it yet.
- **Cancelled harmonics**: an `amplitudes(t, f, n)` function that zeroes harmonic 3 every other
  second works as is.
- **A mistuned harmonic**: a fixed 600 Hz target over a harmonic frame stepped down 1 % at a time.

## Proposed pages

A new gallery group, **Classic demonstrations**, placed after Synthetic sounds. Six pages, each
grouping demonstrations by what they show rather than by disc, and each opening with a short
note on the two sources:

1. **Streams** (sequential grouping). Bregman 1–5, 7–17 (with the amadinda stand-in), 38, 39
   and 41, ASA 19. The six-tone cycle, the gallop, buildup, the hidden melody, timbre and
   spectral-peak streaming, vowels, glides, crossing trajectories, and the same by location.
2. **Fusion** (simultaneous grouping). Bregman 18–24, ASA 1, and comodulation masking release
   (D8). Mistuning, common frequency change, onsets, rhythmic masking release, sine-wave speech,
   vibrato.
3. **Continuity and capture**. Bregman 25–37, ASA 11. The old-plus-new examples: capture,
   continuity through noise, the picket fence, homophonic continuity, residuals, the vowel whose
   third harmonic is captured.
4. **Pitch of complex tones**. ASA 13, 15, 20–23, 25, 26, 32 (secondary beats), 34, with links
   to Chroma, IRN and the phase vocoder's shifted partials.
5. **Levels and masking**. ASA 2–5, 8–10 and 33, with 12 and 14 if wanted.
6. **Two ears**. ASA 37–39 and Bregman 40, with links to Binaural for 36 and to Rooms for 35.

ASA 16, 18 and 31 (scales and stretched tuning) go on the Tuning page instead, and 28 and 29 on
the Timbre page, since those pages already teach what the demonstrations show (D2).

Pages 1–3 together are Bregman's disc, in his order (sequential, simultaneous, old-plus-new,
dichotic), with the dichotic demonstrations split between Streams and Two ears.

Each page follows the gallery's form: a runnable percent-format script in a new
`docs/gallery/demos/` folder, an entry in `build.py`'s `TITLES` and `TOPICS`, a notebook, and a
References section listing only works the page cites.

## Order

Streams first: it holds the best-known demonstrations (the six-tone cycle, the gallop) and is the
page an announcement would point to. Then Fusion, Continuity and capture, Pitch, Levels and
masking, Two ears, then the additions to the Tuning and Timbre pages. One PR per page. Before
each page is merged, Cho gets a private listening page with every new sound.

## Checker

`tools/check_classic_demos.py`, NumPy only, sharing no code with sonore, rebuilds a handful of
stimuli from the booklet numbers and checks the page's versions against them: frequencies and
durations of the gallop and six-tone cycle, the mistuned target staying at its frequency while
the frame moves, the -12 dB flanker components of rhythmic masking release, the phase inversion
in the masking-level-difference signal, the ITD and ILD values, the absence of the 2f1-f2
component from the two-tone signal, and the constant spectrum level across the noise bandwidths
of ASA 2.

## Decisions

- **D1. Scope.** Rebuild every demonstration whose stimulus is synthetic and fully specified;
  use stand-ins where a recording made the point (D5); leave out absolute-threshold staircases
  (ASA 6), procedures (ASA 7 loudness scaling, ASA 17 the answer-keyed difference-limen test)
  and recordings with no good substitute yet (ASA 24, Bregman 6). Recommended.
- **D2. Where the pages go.** A new group, Classic demonstrations, with the six pages above,
  and the scale, tuning and timbre demonstrations added to the existing Music pages. The
  alternative is no new group: each demonstration goes on the existing page closest to it, which
  scatters them and leaves nothing single to point to. Recommended: the new group.
- **D3. Staircases.** Keep the level staircases where the masker is in the same sound (ASA 2, 8,
  9, 10, 38), since the step that disappears depends on the signal-to-masker ratio rather than
  on the reader's volume knob. Drop those against absolute threshold (ASA 6). Shorten each to one
  pass instead of the disc's two. Recommended.
- **D4. Length.** The disc tracks run from 30 s to 4 min. Each example is trimmed to the
  shortest version that keeps the published parameters and the point, usually 5 to 20 s, with a
  longer version only where buildup over time is the point (Bregman 4, 12). Recommended.
- **D5. Stand-ins.** Amadinda (Bregman 7–9, 41): an interlocking pattern written for the page,
  two isochronous cycles on a pentatonic scale, played by a synthetic struck tone, rather than
  the traditional piece's notes. Sine-wave speech (Bregman 23) and the picket fence (31): the
  male talker's sentence, already in the repository. Echoes (ASA 35): a talker sentence in two
  synthetic rooms, played backward. Bell (ASA 28): the booklet's partial frequencies with decay
  times labeled as estimates. The chorale demonstrations (ASA 29, 31) and the melody of ASA 16
  need a public-domain score typed in; Cho may want to choose the piece. The chime (ASA 24) and
  the Telemann excerpt (Bregman 6) are left out for now. Recommended, with the piece open.
- **D6. No library code.** Every stimulus above is plain sonore calls plus page code (a tone
  sequence helper, a gate, a sharp-edged noise band as `harmonic_complex(f0=2, phases="random")`
  over a band). If the same helper ends up on three pages, it is shared from one page-side
  module, not added to sonore. Recommended.
- **D7. Wording of what is heard.** Every perceptual statement is attributed to the booklet or
  the original paper. Where a page has no source for how something sounds, it says what the
  stimulus is and stops. Recommended.
- **D8. Comodulation masking release.** It is on neither disc (Bregman 22's rhythmic masking
  release is a variant of it), but it is one of the classic stimuli. Add it to the Fusion page,
  after Hall, Haggard & Fernandes (1984). Recommended.
- **D9. sonore-infer.** The pages are content only. sonore-infer copies a stimulus's few lines
  when it needs one; no shared stimulus module now. Recommended.

## References

The two booklets were read in full for this note. The original papers are cited from the
booklets and are checked against the publications when each page is written.

- Bregman, A. S. & Ahad, P. A. (1996). *Demonstrations of Auditory Scene Analysis: The
  Perceptual Organization of Sound* (compact disc and booklet). Auditory Perception Laboratory,
  McGill University; distributed by MIT Press. Audio and booklet:
  https://webpages.mcgill.ca/staff/Group2/abregm1/web/downloadsdl.htm
- Bregman, A. S. (1990). *Auditory Scene Analysis: The Perceptual Organization of Sound*. MIT
  Press.
- Houtsma, A. J. M., Rossing, T. D. & Wagenaars, W. M. (1987). *Auditory Demonstrations*
  (compact disc and booklet). Institute for Perception Research (IPO), Eindhoven, and the
  Acoustical Society of America. Booklet:
  https://research.tue.nl/en/publications/auditory-demonstrations
- Hall, J. W., Haggard, M. P. & Fernandes, M. A. (1984). Detection in noise by
  spectro-temporal pattern analysis. *Journal of the Acoustical Society of America* 76, 50–56.
- Slaney, M. Correlograms of the ASA demonstrations:
  https://ccrma.stanford.edu/~malcolm/correlograms/
