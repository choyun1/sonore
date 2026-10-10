# Gallery rewrite

Status: all decisions accepted by Cho (2026-10-10); D9 changed by Cho to keep Organ. Cho asked (2026-10-10) for a large rewrite of the listening
gallery for two reasons: the pages repeat each other, and the female talker reads as an
afterthought. This note maps both problems on the current pages and proposes a structure. The
decisions at the end need Cho's answer before any page changes.

## What is there now

21 pages in five menu groups (`TOPICS` in `docs/gallery/build.py`):

- Stimuli: Synthetic sounds, IRN, Ripples, Textures
- Seeing and changing sound: Seeing speech, Analysis and resynthesis, Vocoder, Modulation
  spectrogram, Hearing a modulation spectrum, Phase vocoder
- Voices: Formants, Cepstrum, Harmonics, Aperiodicity, Changing a voice
- Spatial hearing: Binaural, Reverb, Moving talkers
- Music: Timbre, Tuning and temperament, Organ (a template with no sounds)

## Where the female talker sits

Every recorded-speech page teaches on the male talker (bdl, ARCTIC a0131) and then adds the
female talker (slt, the same sentence) in a late section, usually titled "A higher voice". The
section re-runs the page's pipeline on slt and remarks on what changed.

| Page | Female material | Placement |
| --- | --- | --- |
| Seeing speech | f1 | own section, second to last |
| Vocoder | cf0, cf8, cf8p | own section, after the male teaching |
| Modulation spectrogram | sv1 | own section |
| Phase vocoder | p4, p5 | own section |
| Cepstrum | c6, c7 | own section, after the male teaching |
| Harmonics | hv1, hv2 | own section near the end, with a `resynthesize()` copy of the page |
| Aperiodicity | ap14 to ap17 | own section near the end |
| Formants | fw4, fw5 | own section ("Female vowels"), mid-page |
| Changing a voice | vc13 to vc16 | own section; the female talker is the conversion target |
| Reverb | r3 | one extra demo, no section |
| Moving talkers | m8, m9 | own section; the page intro still says "Three male talkers" |
| Synthetic sounds | k2 | woven into the long-term spectrum section |
| Timbre | tb6 | woven in (female formants on one note) |
| Moving talkers, cocktail party | cp1 to cp5 | woven in (mixed LibriSpeech casts) |

The pages also disagree about how far apart the two voices are. Measured with `so.f0_track` on
the two committed recordings, the median F0 is 119.4 Hz for bdl and 182.6 Hz for slt, a ratio of
1.53 (about 7.4 semitones, a little more than a fifth). The pages say:

- "half as high again" (Cepstrum, Harmonics) and "about a fifth higher" (Phase vocoder): right.
- "about half an octave to an octave higher" (Seeing speech, Aperiodicity): a general range, not
  this pair.
- "roughly an octave higher", "harmonics twice as far apart" (Modulation spectrogram): wrong for
  this pair.
- Aperiodicity's synthetic vowel at 230 Hz is "twice the pitch" of its 115 Hz male vowel: true of
  that vowel, but it reads as a claim about the talkers.

So the female voice is added after the fact on about ten pages, each time with its own setup and
its own statement of the difference.

## What is repeated

Concepts explained on more than one page (owner proposed in D5):

| Concept | Pages that explain it |
| --- | --- |
| Source and filter | Formants, Cepstrum, Aperiodicity, Harmonics, Changing a voice, Seeing speech |
| Liftered cepstral envelope | Cepstrum (twice), Harmonics, Changing a voice |
| Cepstral F0 vs `so.f0_track` vs Harvest, octave errors | Seeing speech, Cepstrum, Harmonics, Changing a voice |
| Pitch moved with formants fixed vs phase vocoder moving both | Harmonics (hm2), Changing a voice (vc7), Phase vocoder (p3) |
| Monotone and pitch range | Harmonics (hm1), Changing a voice (vc9) |
| Breathiness and aspiration noise | Formants, Aperiodicity, Changing a voice |
| Wider harmonic spacing samples the formants sparsely | Formants, Seeing speech, Cepstrum, Harmonics, Aperiodicity |
| Female formants higher, shorter tract | Formants, Changing a voice, Phase vocoder |
| Klatt "hod" vowel and glottal source setup | Formants, Aperiodicity |
| MFCC recipe | Cepstrum, Changing a voice |
| STFT invertibility and overlap-add | Seeing speech ("Nothing is lost"), Analysis and resynthesis, Phase vocoder |
| ERB filterbank and gammatone | Seeing speech, Analysis and resynthesis |
| Wideband striations as glottal pulses | Seeing speech, Vocoder |
| Modulation spectrum, defined | Ripples, Hearing a modulation spectrum, Modulation spectrogram, Textures (a different per-band construction) |
| IRN vs ripples | IRN and Ripples end on a near-identical paragraph |
| Beats between near partials | Synthetic sounds, Tuning, Organ |
| Distance: inverse square law, direct to reverberant ratio | Reverb ("Farther away"), Moving talkers ("Coming closer") |
| Formants as timbre | Timbre ("Vowels are timbres"), Formants |

Repeated setup code: the same `finish()` and `plt.rcParams` lines on 23 pages; the sentence and
its Harvest track loaded on 9; `so.f0_track` on slt run on 6; wideband spectrogram panels written
four ways; `synth_ir` rooms in Reverb and Moving talkers; the same bdl/rms file list in Synthetic
sounds and Moving talkers.

Some content sits in the wrong group: binaural beats are on Synthetic sounds, not Binaural (which
never mentions them); the male and female long-term spectra are on Synthetic sounds; Organ has no
sounds yet but is a full menu entry.

## Proposed structure

Six groups (five today), simple to elaborate as now. Pages marked new or merged; the rest keep their file and
are trimmed.

1. Stimuli: Synthetic sounds (beats and roughness, tone sequences, band-limited waveforms,
   speech-shaped noise), IRN, Ripples, Textures.
2. Analysis and resynthesis: Analysis and resynthesis (takes "Nothing is lost" from Seeing
   speech), Phase vocoder.
3. Voices: Two talkers (new, the reference page), Seeing speech (moved here, see D14), Formant
   synthesis, Pitch (new, from the tracker
   parts of Cepstrum, Harmonics and Seeing speech), Spectral envelope (Cepstrum plus MFCCs),
   Source and aperiodicity, Rebuilding and changing a voice (Harmonics and Changing a voice
   merged).
4. Envelopes and modulation: Vocoder, Modulation spectrogram, Hearing a modulation spectrum.
5. Spatial hearing: Binaural (takes binaural beats), Rooms (Reverb renamed, owns distance),
   Moving talkers.
6. Music: Timbre, Tuning and temperament, Organ (kept, D9).

That is 22 pages in the menu, one more than now (two are new, two merge into one), but the count is not the aim: the aim is one owner per concept
and both talkers in the main line of every speech page.

## Decisions

**D1. How the two talkers appear.** Options:
(a) Pairs throughout: every speech demo and figure that uses a recording is made for both
talkers, shown side by side (two columns on wide screens, stacked on phones), and the prose
discusses both where the concept is taught. No "A higher voice" sections remain.
(b) A talker switch: each demo has a male/female toggle in the player; the prose talks about
whichever is the point.
(c) Alternate leads: each page uses one talker, chosen by the point (female where higher pitch
is what the page is about, male elsewhere), with the other talker in a short comparison.
Recommendation: (a). It makes the comparison the teaching, not an appendix. It roughly doubles
speech media; (b) costs the same media plus player work, and hides one voice by default.

**D2. Which talker comes first.** In a pair, which talker is on the left or first in the prose.
Recommendation: the female talker first on pages where the higher pitch is what breaks or
stresses the method (Spectral envelope, Pitch, Source and aperiodicity, Vocoder), the male
talker first elsewhere. Alternative: always female first, or always alphabetical.

**D3. A "Two talkers" reference page.** One page opens the Voices group: both recordings, their
measured F0 medians and ranges, their long-term spectra (k2 moves here from Synthetic sounds),
their formant ratios, and the shared loading code. Every other page states the difference only
by linking here, so it is said once and measured once (a checker in `tools/` computes the
numbers the page prints). Recommendation: yes.

**D4. Shared setup code.** The code is on the page so a reader can paste it into a notebook and
get exactly the same sound and figure (Cho, 2026-10-10). So the setup has to be on every page in
full, not behind an import. Proposal: the setup is written once in `docs/gallery/common.py`, and
every page opens with it in full as a setup cell, so what is shown is what ran. As built, the
copies live in the page scripts themselves (`tools/sync_gallery_setup.py` writes them and a docs
test checks them): a script that imported or `exec`ed the file would not run on its own once
pasted, and linting could not see its names. Each page also offers its whole code as one download (a `.py` or
`.ipynb` built from the same cells), so a reader does not copy cell by cell. Two things today
stop a paste from reproducing outside a clone of the repository: pages load recordings from
relative paths (`docs/speech/...`), and some setup lives only on the page. The setup cell would
load recordings the way the Colab tutorial's setup does, from a URL, so a pasted page runs in a
fresh notebook. Recommendation: yes to all three parts.

**D5. One owner per concept.** Each concept in the table above is taught on one page and linked
from the others. Proposed owners: source and filter, Klatt setup, female formants: Formant
synthesis. Pitch trackers and octave errors: Pitch. Liftered envelope and MFCCs: Spectral
envelope. Sparse harmonic sampling: Two talkers (it is the one thing the pair shows on every
page). Breathiness: Source and aperiodicity. Pitch with envelope fixed vs phase vocoder,
monotone and range: Rebuilding and changing a voice. STFT invertibility and filterbank:
Analysis and resynthesis. Modulation spectrum definition: IRN and ripples. Beats: Synthetic
sounds. Distance: Rooms. Recommendation: accept the list, and correct it where Cho disagrees.

**D6. Voices group split.** Five pages become six, organized by the parts of a voice model
(pitch, envelope, source noise, synthesis, change) instead of by method (cepstrum, harmonics).
Alternative: keep the five pages and only remove repeats. Recommendation: the split; it is where
most repeats are.

**D7. IRN and Ripples.** Keep both pages, delete the repeated closing paragraph from IRN, and let
Ripples own the definition of the modulation spectrum. Alternative: merge them into one page.
Recommendation: keep both; the two are different enough to stand alone.

**D8. Move binaural beats to Binaural.** Recommendation: yes, with a redirect from the old
anchor.

**D9. Organ stays in the menu.** Decided by Cho (2026-10-10): keep the page as a promissory note for the stop sounds to come.

**D10. Moving talkers' cast.** "Three talkers" uses a mixed cast from the start (for example a
female target with male maskers and the reverse), so the other-sex release (Brungart 2001) is
part of the main demos rather than "A different voice". Recommendation: yes.

**D11. Old links.** Merged or moved sections keep working through the existing anchor redirect
script in each page and a stub page for any removed URL. Recommendation: yes.

**D12. Images.** A rewrite regenerates most images. Images built in this cloud container differ
slightly from Cho's local builds (matplotlib), so until now only new images were committed from
here. Options: (a) accept cloud-built images for rewritten pages; (b) Cho rebuilds locally
before each merge. Recommendation: (a) for rewritten pages only, since the old images go away
anyway; the README images stay untouched.

**D13. Rollout.** One PR per group, Voices first (with Seeing speech in it, D14) (it carries most of the female-talker work and
most repeats), then Time and frequency, Spatial, Stimuli, Music. Each PR fixes the pitch
statements it touches to the measured ratio. Recommendation: yes.

**D14. Seeing speech joins Voices.** Cho suggested it (2026-10-10). The page teaches
time-frequency analysis, but always on speech, and its motivation is measuring a voice (pitch,
envelope, glottal pulses). In Voices it comes right after Two talkers, so the pair is introduced
and then seen; the two spectrograms show the tradeoff differently (the female talker's harmonics
are resolved by a shorter window). The old group keeps Analysis and resynthesis and Phase
vocoder. Recommendation: yes.

**D15. A prose pass on every page.** Each group's PR also corrects and clarifies the prose it
touches: wrong or inconsistent claims (the pitch ratios above), uncited claims (for example
"octave errors are the most common", the ba/da/ga transition cues), statements the demo does not
show (Harmonics says the contour marks a question, but the sentence is a statement), and passages
that are hard to follow. Every correction is listed in the PR description with what was checked,
and a new number is measured by a checker or labeled an estimate. Citations get the same pass: each
one the page keeps is checked against the work itself (authors, year, title, journal, and the
claim it supports), linked by DOI or a stable URL, listed in the page's References only if the
text cites it, and present in the README References (the existing docs test checks the last
part). Recommendation: yes, folded
into the group PRs rather than a separate sweep, so each page is rewritten once.

**D16. Cross-links between demos.** As the Synthetic sounds page and Tuning and temperament now
do for the just major third (b3 and tt1), a demo that illustrates a point made on another page
links to that page's demo anchor (`page.html#d-KEY`), and the other page links back, in the
sentence that makes the point. This is in addition to D5: D5 says where an idea is taught, D16
connects sounds that show the same thing. Candidates found in the audit: beats (Synthetic sounds
b1/b2, Tuning tt1/tt8, Organ celeste); binaural beats and interaural correlation (b4, Binaural
07/08); rippled noise and ripples (IRN 09, Ripples 01); speech-shaped noise and the modulation
spectrum (k1, Hearing a modulation spectrum mt1); wideband striations and envelope pitch (Seeing
speech 27, Vocoder cp8); pitch moved with formants fixed (Harmonics hm2, Changing a voice vc3,
Phase vocoder p3); breathy voice (Formants fq5, Aperiodicity ap3); vowels as timbre (Timbre tb6,
Formants fw1); distance (Reverb r2, Moving talkers m5); envelopes and fine structure (Vocoder,
Hearing a modulation spectrum mt5). The existing test that checks gallery anchors covers the new
links. Recommendation: yes, added page by page in each group PR.

## Not changing

Demo sounds that are not speech, the gallery's look, the sidebar and menus (only their entries),
the API reference, the README. No library code changes are expected; if a page needs one, it gets
its own note first.
