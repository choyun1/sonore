# Chroma, Shepard tones and the constant-Q filter bank

Cho asked on 2026-10-14 what sonore is missing for music analysis. Most music
software (librosa, madmom, essentia) is about labeling recordings: beats,
chords, keys, notes. sonore leaves that alone. Three pieces fit sonore
because they are about the sound, not about a label, and Cho asked for all
three:

1. **A constant-Q filter bank**, one band per semitone, that gives the sound
   back exactly.
2. **Chroma**: the energy in each of the 12 pitch classes, with the octave
   folded away. It is a one-way view, and the octave height is what it drops.
3. **Shepard tones and the Risset glide**: sounds with a pitch class and no
   octave height, the classic stimuli built from the same folding.

Cho's ruling, 2026-10-14: Shepard and Risset tones are not classes or new
functions. They are recipes for `so.harmonic_complex`, because they are
just particular choices of harmonics and amplitudes. This document proposes
the rest. It adds no library code; the decisions at the end come first.

Status: Cho accepted D1, D2, D5 and D6 as recommended on 2026-10-14. D4
is Cho's ruling. D3 accepted by Cho the same day, with the spread over
octaves as an option (below).

## How the claims are verified

- **[check]**: printed by `tools/check_chroma_claims.py` (K1 to K6, K8). It uses
  NumPy only, shares no code with sonore, and runs in about 1 s.
- **[librosa]**: printed by `tools/crosscheck_cqt_librosa.py` (K7), run with
  librosa 1.0.0. librosa is not a sonore dependency.
- **[sonore]**: measured with sonore on main (0.7.0 plus unreleased work),
  with scripts not committed. Each measurement is described where it
  appears.
- **[math]**: follows in one line from the formulas shown.

## 1. The constant-Q filter bank is already in sonore

A constant-Q transform splits a sound into bands whose width is a fixed
fraction of their center frequency, so every band covers the same musical
interval (Brown, 1991). sonore's cosine filter bank on the cents scale is
one:

```python
bank = so.cosine_filterbank(f_lo=55, f_hi=7040, scale="cents", spacing=100)
```

- **Constant Q.** Each band is a half-cycle cosine in log frequency, zero at
  its neighbors' centers. Its half-power points are half a spacing away on
  either side, so Q = 1 / (2^(s/2) − 2^(−s/2)) for a spacing of s octaves:
  17.31 at one band per semitone, 51.94 at three [math, check K1]. Measured
  on sonore's bank at 22.05 kHz, Q runs from 17.31 to 17.47 over the 83
  bands; the spread comes from the frequency grid the responses were
  sampled on [sonore].
- **Exact.** The squared responses sum to 1 at every frequency, edges
  included, so filtering twice and adding the bands gives the sound back.
  On 2 s of white noise, the independent build gives a relative error of
  9e-15 [check K2]. sonore's bank gives 4e-16 [sonore].
- **The cost is redundancy.** The bank is undecimated: every band keeps
  every sample, 85 real values per input sample here [check K2]. That is
  the price of being exact with no further design.

librosa's constant-Q transform is decimated and only approximately
invertible. On band-limited noise its inverse comes back with a relative
error of 0.80 at 12 bins per octave and a hop of 256 samples, and about
0.02 at 36 or 48 bins per octave with short hops (64 or 32 samples). That's
after fitting the best overall gain, because the level does not come back
either (gain 0.41 to 1.00) [librosa K7]. Exact, decimated constant-Q
transforms exist (nonstationary Gabor frames, Velasco et al., 2011;
Holighaus et al., 2013). sonore does not have one, and this proposal does
not add one (D5).

## 2. Chroma

Chroma (Fujishima, 1999, who called it a pitch class profile) is computed
in three steps:

1. Split the sound by semitone: the filter bank above.
2. Take each band's power envelope and average it over time windows.
3. Add every band of the same pitch class together, so C2 + C3 + C4 + …
   becomes one row for C. That leaves 12 rows instead of about 80.

**What it drops.** Octave height: a low C and a high C land in the same
row. It also drops everything within a band, such as timbre, phase and the
exact tuning inside a semitone. Many sounds share one chroma, so it is a
view, not a frame.

**What it adds that is not there.** Overtones land on other pitch classes.
The third harmonic of C is a G, and the fifth is an E 14 cents flat. For a
C major triad (C4, E4, G4) of harmonic tones with amplitudes 1/n and 10
harmonics, C, E and G hold 29%, 30% and 32% of the power. The other 10%
lands mostly on B and D, from overtones [check K3]. Chroma is a summary of
partials, not a list of notes.

**Its way back to a sound.** sonore's views take back what they dropped as
arguments to `to_sound` (`Spectrum.to_sound` takes a carrier for the
phase). Chroma drops octave height, and a Shepard tone is a sound with a
pitch class and no height. So `Chroma.to_sound` plays each pitch class's
row as a Shepard tone, and the missing height comes in as the bell-shaped
level over log frequency the Shepard tones use (D3). Its output is a
recipe of `harmonic_complex` calls, as in section 3.

The reason it is a Shepard tone and not some other sound with that pitch
class: every partial of a Shepard tone is an octave of one pitch class, so
all of its power folds into that one row. Through the semitone bank, a
steady Shepard tone on C puts all of its chroma in C (share 1 to six
digits), and a harmonic tone C4 puts 86% there, the rest leaking through its
overtones [check K8]. So the chroma of the sound that `to_sound` makes is the
chroma it was given, up to the smoothing of the time windows, which is not
checked yet.

## 3. Shepard tones and the Risset glide, as recipes

**Shepard tone** (Shepard, 1964). Its partials are an octave apart, so
the harmonic numbers are powers of two. Their level is a fixed bell shape
over log frequency:

```python
center, width = np.log2(500), 1.0  # bell center [log2 Hz] and width [octaves]

def bell(t, f):
    return np.exp(-0.5 * ((np.log2(f) - center) / width) ** 2)

c = so.note_to_freq("C4") / 2**6  # C four octaves below the lowest audible C
tone = so.harmonic_complex(1.0, 44100, c, harmonics=2 ** np.arange(11), amplitudes=bell)
```

A Shepard tone and the same tone an octave up have the same partials,
except at the ends of the range. Starting at C1 with every partial kept up
to Nyquist, they share 8 partials, and the ones not shared carry 1e-7 of
the power [check K4]. A semitone step is `c * 2 ** (1/12)`. Twelve steps
come back to the start, which is the endless scale.

**Nothing in the spectrum says which of two Shepard tones is higher.** The
power-weighted mean log frequency of the 12 pitch classes varies by less
than 0.001 octave [check K5]. Two tones a tritone apart are the classic
ambiguous case. Listeners disagree about which way the step goes, and the
disagreement depends on the pitch class (Deutsch, 1986). That finding is
cited, not tested here. If relative pitch is ever modeled, that belongs in
sonore-infer.

**Risset glide** (Risset, 1969). The same recipe with a rising F0 contour.
The lowest partial starts five octaves below 20 Hz, so new partials keep
entering from below while old ones fade out at the top:

```python
t = np.linspace(0, 8, 200)
glide = (t, 20 / 2**5 * 2 ** (t / 2))  # one octave every 2 s
sound = so.harmonic_complex(8, 44100, glide, harmonics=2 ** np.arange(20), amplitudes=bell, f_max=22050)
```

In sonore, that call gives a level within 0.1% from second to second and a
power-weighted mean log frequency that stays within 0.002 octave of the
bell's center (8.966 to 8.968, for 8.966 at 500 Hz) [sonore: RMS and the
mean over semitone bands from 20 Hz to 20 kHz, in 1 s windows]. The
independent build agrees: level ratio 1.001, spread 0.0006 octave
[check K6]. `harmonic_complex` already does everything the glide needs, so
nothing is added.

## Decisions

**D1. Chroma as a view. Accepted by Cho, as recommended: `so.chroma(sound, ...)` returning
`Chroma`, a `View` in `views/chroma.py`.** It builds the semitone bank,
takes power envelopes, averages them over time windows and folds the
octaves. `Chroma` holds the array (time windows × 12 × channels), the times,
and the reference pitch. It also has `plot` (12 rows over time, named C to
B) and the `discards` sentence: "octave height and everything within a
semitone band". The alternative, a method on `Subbands`, would only work
on a bank spaced in whole semitones, and the bank would have to know about
pitch classes.

**D2. Defaults. Accepted by Cho, as recommended (estimates, to settle when built):** one band
per semitone from A1 (55 Hz) to A8 (7040 Hz), reference A4 = 440 Hz (an
argument, for recordings tuned elsewhere), 0.05 s time windows. Power is
kept as measured, with no normalization per window. Normalizing hides how
loud the window was, so `plot` can offer it and the data stays as measured.
A tuning offset estimated from the sound is left out: it is a guess about
the recording, not part of the view.

**D3. `Chroma.to_sound`. Accepted by Cho: yes, as Shepard tones by default,
with the spread over octaves as an option.** One argument, `octaves`, says
how each pitch class's energy spreads across octaves, which is what the view
dropped: the Shepard bell by default (no height), a note or frequency such
as `"C4"` for a narrow bell there (every pitch class near that octave), or
a function of frequency for any other spread. All of them come back to the
same chroma, because every partial is still an octave of its pitch class.
Also `fs`. Each
pitch class's row becomes the time-varying level of one Shepard-tone recipe
(one `harmonic_complex` call per pitch class, summed). That shows on one
page what chroma keeps and what it drops.

Why a Shepard tone (Cho asked): it is the sound whose chroma comes back
unchanged (section 2, K8), and its one free choice, the bell, is exactly
what the view dropped. It is not the only sound that comes back unchanged.
Pure tones in one octave do too, but they put back a height the view never
had. Noise bands for each pitch class across octaves come back unchanged
only on average, and differ from run to run. The Shepard tone is the
deterministic choice that invents no height. The alternative is no route
back: `to_sound` refuses and names Shepard tones as a recipe.

An option does not break the rule for `to_sound`: the rule is that its
arguments are what the view dropped, not that there is one answer
(`Spectrum.to_sound` takes `carrier="noise"`, `"minimum"` or a Sound).

No separate helper that picks an octave for any harmonic complex. For a
Shepard tone, where the energy sits is the height. For an ordinary harmonic
tone, the pitch stays at its F0 wherever the energy sits (the missing
fundamental), and moving the energy only makes it brighter or duller. One
helper would then pick an octave for some sounds and change only the timbre
of others. Reweighting partials by a level over log frequency is already
`harmonic_complex`'s `amplitudes=`, and the Shepard recipe with a narrower
bell picks an octave.

**D4. Shepard and Risset recipes. Settled by Cho: recipes, no new code.**
They live in the gallery page and in one line of `harmonic_complex`'s
docstring that points to it. Proposed for that line: "octave-numbered
harmonics with a bell-shaped level over log frequency make a Shepard tone;
see the gallery".

**D5. Constant-Q. Accepted by Cho, as recommended: no new function.** The gallery shows that
`cosine_filterbank(scale="cents", spacing=100)` is an exact constant-Q
filter bank, gives its Q, and says what it costs (undecimated). A
decimated exact version (nonstationary Gabor frames) stays where it already
is, on the README roadmap, as separate work.

**D6. Gallery. Accepted by Cho, as recommended: a new Music page, "Pitch class and height",
after Tuning and temperament.** Sections:

1. A constant-Q filter bank: the bank, its Q, the exact round trip.
2. Folding octaves: chroma of a C major triad and of a short melody, with
   the overtone leakage shown.
3. Shepard tones: the endless scale, and two tones a tritone apart (cited
   to Deutsch; the page makes no claim of its own about what is heard).
4. The Risset glide.
5. Back from chroma: a melody's chroma played as Shepard tones (if D3).

The Music group would then run Timbre, Tuning and temperament, Pitch class
and height, Pipe organ.

## Not proposed

Beat and tempo tracking, chord and key finding, note transcription and
source separation. They label recordings, librosa and madmom already do
them well, and the good ones are trained models rather than derivations a
notebook can show. Harmonic/percussive separation (median filtering a
spectrogram, then a mask) would fit as a mask on a frame. It is a possible
later page, not part of this one.

## References

DOIs and page ranges are from memory and have not been checked against the
papers. They will be checked before the page is built.

- Brown, J. C. (1991). Calculation of a constant Q spectral transform.
  *J. Acoust. Soc. Am.* 89(1), 425–434.
- Deutsch, D. (1986). A musical paradox. *Music Perception* 3(3), 275–280.
- Fujishima, T. (1999). Realtime chord recognition of musical sound: a
  system using Common Lisp Music. *Proc. International Computer Music
  Conference*, 464–467.
- Holighaus, N., Dörfler, M., Velasco, G. A., & Grill, T. (2013). A framework
  for invertible, real-time constant-Q transforms. *IEEE Trans. Audio,
  Speech, and Language Processing* 21(4), 775–785.
- Risset, J.-C. (1969). Pitch control and pitch paradoxes demonstrated with
  computer-synthesized sounds. *J. Acoust. Soc. Am.* 46, 88 (abstract).
- Shepard, R. N. (1964). Circularity in judgments of relative pitch.
  *J. Acoust. Soc. Am.* 36(12), 2346–2353.
- Velasco, G. A., Holighaus, N., Dörfler, M., & Grill, T. (2011).
  Constructing an invertible constant-Q transform with nonstationary Gabor
  frames. *Proc. DAFx-11*, 93–99.
