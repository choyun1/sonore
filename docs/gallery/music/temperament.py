"""Tuning and temperament: reference pitch, just intervals, commas and four tunings of one tune.

This script is the gallery page https://choyun1.github.io/sonore/gallery/temperament.html:
docs/gallery/build.py runs it cell by cell from the repository root and shows each
cell's code beside what it made. Run it yourself from the repository root,

    python docs/gallery/music/temperament.py

or a cell at a time ("# %%" starts a cell in VS Code, Spyder and Jupytext).
"""

# %% [markdown]
# # Tuning and temperament
#
# Two notes sound smooth together when their partials fall on one another, which happens when
# their frequencies stand in a ratio of small whole numbers. Twelve notes to the octave cannot
# all do that at once, so every keyboard tuning is a compromise, and the compromises have
# names. This page plays them, and draws each pair of notes as a Lissajous figure, which stands
# still when the ratio is exact and turns when it is not.
#
# - [Intervals are ratios](#h-intervals-are-ratios): a just fifth and third against tempered
#   ones, heard as beats and seen as a figure that turns.
# - [The commas](#h-the-commas): why twelve notes cannot keep every fifth and third just.
# - [One tune, four tunings](#h-one-tune-four-tunings): Twinkle, Twinkle, Little Star in equal
#   temperament, just intonation, Pythagorean tuning and quarter-comma meantone.
# - [Where just intonation breaks](#h-where-just-intonation-breaks): one chord that just
#   intonation fixed on C cannot tune.
# - [A reference pitch is a choice](#h-a-reference-pitch-is-a-choice): the tune at
#   A4 = 440 Hz and at 432 Hz, and a short history.

# %% [markdown]
# ## Code the examples share
#
# Every note is a harmonic complex tone of eight partials with amplitudes falling as $1/n$, so
# that the partials of two notes can meet and beat; the sounds are at 22.05 kHz. An interval is
# measured in cents, 1200 to the octave: $1200 \log_2 r$ for a frequency ratio $r$
# (`so.ratio_to_cents`). A tuning here is a table of the seven note names of C major as ratios
# to C, and every tuning keeps C4 where equal temperament with A4 = 440 Hz puts it
# (`so.note_to_freq`), so that the tunings differ only in the other notes.
#
# Each sound is drawn as a spectrogram on a logarithmic frequency axis, and under it each note
# as its distance in cents from the same note in equal temperament with A4 = 440 Hz. The square
# figure beside it is the Lissajous figure of the lowest note (across) against the highest
# (up): two sine waves at their fundamentals, drawn over the last few hundredths of a second as
# the sound plays. Inner notes of a chord are heard but not drawn.

# %%
import matplotlib.pyplot as plt
import numpy as np

import sonore as so
from sonore.plotting import plot_lissajous

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 100})
FS = 22050
C4 = so.note_to_freq("C4")
PARTIALS = np.arange(1, 9)

# The seven notes of C major as frequency ratios to C, in four tunings. Quarter-comma meantone
# narrows every fifth by a quarter of the syntonic comma, so that four fifths make a just
# major third: its fifth is the fourth root of 5.
MEANTONE_FIFTH = 5**0.25
TUNINGS = {
    "equal temperament": {
        name: 2 ** (steps / 12) for name, steps in zip("CDEFGAB", (0, 2, 4, 5, 7, 9, 11), strict=True)
    },
    "just intonation": {"C": 1, "D": 9 / 8, "E": 5 / 4, "F": 4 / 3, "G": 3 / 2, "A": 5 / 3, "B": 15 / 8},
    "Pythagorean": {"C": 1, "D": 9 / 8, "E": 81 / 64, "F": 4 / 3, "G": 3 / 2, "A": 27 / 16, "B": 243 / 128},
    "quarter-comma meantone": {
        "C": 1,
        "D": MEANTONE_FIFTH**2 / 2,
        "E": 5 / 4,
        "F": 2 / MEANTONE_FIFTH,
        "G": MEANTONE_FIFTH,
        "A": MEANTONE_FIFTH**3 / 2,
        "B": MEANTONE_FIFTH**5 / 4,
    },
}


def pitch(note, tuning, c4=C4):
    """Frequency of a note name such as "G4" in one of TUNINGS, with C4 at ``c4``."""
    return c4 * TUNINGS[tuning][note[0]] * 2.0 ** (int(note[1:]) - 4)


def tone(f0, duration):
    """One note: eight harmonics falling as 1/n, with 20 ms ramps."""
    return so.harmonic_complex(duration, FS, f0, harmonics=PARTIALS, amplitudes=1 / PARTIALS, ramp=20e-3)


def finish(snd):
    """How every sound in the gallery is played: 5 ms ramps, RMS 0.1, peak at most 0.95."""
    snd = snd.ramp(5e-3).normalize(rms=0.1)
    return snd.normalize(peak=0.95) if snd.peak > 0.95 else snd


def play(events, total):
    """Mix ``events``, rows of (start [s], duration [s], frequency [Hz]), into one sound."""
    data = np.zeros(int(round(total * FS)))
    for start, duration, f0 in events:
        note = tone(f0, duration).data[:, 0]
        first = int(round(start * FS))
        data[first : first + len(note)] += note
    return so.Sound(data, FS)


def figure_notes(chords, label=None):
    """Lissajous rows (start, end, lowest, highest, label) for chords given as
    (start, duration, frequencies)."""
    rows = []
    for start, duration, freqs in chords:
        low, high = min(freqs), max(freqs)
        text = label(low, high) if label else f"{high / low:.3f} : 1"
        rows.append((start, start + duration, low, high, text))
    return rows


def show(snd, events, fmin=80, fmax=3000, reference=440.0):
    """Spectrogram on a log frequency axis, and each note's distance in cents from equal
    temperament at A4 = ``reference``. Returns the figure and the panels the playhead follows."""
    fig, (ax_s, ax_c) = plt.subplots(
        2, 1, figsize=(10, 5.4), sharex=True, height_ratios=[1.7, 1], layout="constrained"
    )
    stft = so.STFT(snd, 80e-3, 10e-3)
    keep = (stft.f >= fmin) & (stft.f <= fmax)
    level = stft.db[0][keep]
    ax_s.pcolormesh(stft.t, stft.f[keep], level, vmin=level.max() - 70, vmax=level.max(), cmap="magma")
    ax_s.set(yscale="log", ylabel="Frequency [Hz]", title="Spectrogram (Hann 80 ms)")
    for start, duration, f0 in events:
        steps = 12 * np.log2(f0 / reference) + 69  # MIDI note number, equal temperament at A4 = 440
        offset = 100 * (steps - np.round(steps))
        ax_c.plot([start, start + duration], [offset, offset], lw=3, solid_capstyle="butt", color="C0")
    ax_c.axhline(0, color="0.5", lw=0.8)
    ax_c.set(
        xlim=(0, snd.duration),
        ylim=(-36, 26),
        xlabel="Time [s]",
        ylabel="Cents",
        title="Each note against equal temperament, A4 = 440 Hz",
    )
    ax_c.grid(ls=":")
    return fig, [ax_s, ax_c]


# %% [markdown]
# ## Intervals are ratios
#
# The partials of a note at $f$ lie at $f, 2f, 3f, \ldots$ Raise a second note a fifth, to
# $\tfrac{3}{2}f$, and its second partial lands exactly on the first note's third; raise it a
# major third, to $\tfrac{5}{4}f$, and its fourth partial lands on the first note's fifth. Those
# are just intervals: the shared partials coincide, and nothing beats. Equal temperament makes
# every semitone the twelfth root of two, 100 cents, so its fifth is
# {{ f"{700 - so.ratio_to_cents(3 / 2):.2f}" }} cents narrower than 3:2, and its major third
# {{ f"{400 - so.ratio_to_cents(5 / 4):.2f}" }} cents wider than 5:4. Near-coinciding partials
# beat at the difference of their frequencies, which the cell prints for intervals above
# A3 = 220 Hz.

# %%
A3 = 220.0
for name, steps, just_ratio, low_n, high_n in [
    ("fifth", 7, 3 / 2, 3, 2),
    ("major third", 4, 5 / 4, 5, 4),
]:
    upper = A3 * 2 ** (steps / 12)
    detune = 100 * steps - so.ratio_to_cents(just_ratio)
    beat = abs(high_n * upper - low_n * A3)
    print(f"equal {name}: {upper:.2f} Hz, {detune:+.2f} cents from just {just_ratio:.4g}")
    print(f"    partial {low_n} of A3 and partial {high_n} of the upper note beat at {beat:.2f} Hz")

# %% [markdown]
# A Lissajous figure plots one sine wave against another. When the frequencies stand at
# exactly 3:2 the curve closes on itself and stays put; when they are slightly off, the figure
# slowly turns through every shape it can take and starts again. For two frequencies near 3:2
# that takes $1/|2f_{upper} - 3f_{lower}|$ seconds, the reciprocal of the beat rate of the
# nearest shared partials: the figure turns once per beat. The still figures below (half a
# second of two pure tones, `plot_lissajous`) show the same thing as a smear.

# %% [figure tt0] Lissajous figures, half a second each
pairs = [
    ("just fifth, 3:2", A3 * 3 / 2),
    ("equal fifth", A3 * 2 ** (7 / 12)),
    ("just major third, 5:4", A3 * 5 / 4),
    ("equal major third", A3 * 2 ** (4 / 12)),
]
fig, axes = plt.subplots(1, 4, figsize=(10, 2.9), layout="constrained")
for ax, (name, upper) in zip(axes, pairs, strict=True):
    t = np.arange(int(0.5 * FS)) / FS
    stereo = so.Sound(np.stack([np.sin(2 * np.pi * A3 * t), np.sin(2 * np.pi * upper * t)], axis=1), FS)
    plot_lissajous(stereo, ax, lw=0.3)
    ax.set(title=name, xlabel="A3, 220 Hz", ylabel=f"{upper:.2f} Hz")

# %% [about]
# A3 with a fifth above it, just and then equal-tempered, then a major third, just and then
# equal-tempered, three seconds each. The just intervals are smooth; the tempered fifth beats
# slowly, under once a second, and the tempered third flutters at almost 9 beats a second. The
# figure stands still for the just intervals and turns at the beat rate for the tempered ones.

# %% [demo tt1] Just and tempered intervals over A3
names = [name for name, _ in pairs]
uppers = [upper for _, upper in pairs]
chords = [(3.2 * i, 3.0, (A3, upper)) for i, upper in enumerate(uppers)]
events = [(start, duration, f) for start, duration, freqs in chords for f in freqs]
sound = finish(play(events, 12.6))
fig, playhead = show(sound, events)
lissajous = {
    "notes": [(*row[:4], name) for row, name in zip(figure_notes(chords), names, strict=True)],
    "window": 0.02,
    "xlabel": "A3",
    "ylabel": "upper note",
    "title": "Lowest note against highest",
    "start": 4.0,
}

# %% [markdown]
# ## The commas
#
# If fifths and thirds can be just, why not tune every one of them so? Because twelve notes
# cannot hold them all. Stack twelve just fifths from C and you come back to a C that
# overshoots seven octaves by the Pythagorean comma, $3^{12}/2^{19}$, or
# {{ f"{so.ratio_to_cents(3**12 / 2**19):.2f}" }} cents. Stack four just fifths, C–G–D–A–E, and the E
# overshoots a just major third two octaves up by the syntonic comma, 81:80, or
# {{ f"{so.ratio_to_cents(81 / 80):.2f}" }} cents. Each tuning decides where those commas go, and
# the cell prints the result for the four tunings of this page: the fifths and major thirds of
# C major's three main chords, and the fifth and minor third of D minor.

# %%
intervals = [("C", "G"), ("F", "C"), ("G", "D"), ("D", "A"), ("C", "E"), ("F", "A"), ("G", "B"), ("D", "F")]
print(f"{'cents':24s}" + "".join(f"{low + '-' + high:>6s}" for low, high in intervals))
for tuning, table in TUNINGS.items():
    ratios = [table[high] / table[low] * (2 if table[high] < table[low] else 1) for low, high in intervals]
    sizes = so.ratio_to_cents(ratios)
    print(f"{tuning:24s}" + "".join(f"{size:6.1f}" for size in sizes))
just_sizes = so.ratio_to_cents([3 / 2] * 4 + [5 / 4] * 3 + [6 / 5])
print(f"{'just intervals':24s}" + "".join(f"{size:6.1f}" for size in just_sizes))

# %% [markdown]
# Equal temperament spreads the Pythagorean comma evenly, so every fifth is 700 cents and
# every key is alike, and the thirds pay. Pythagorean tuning keeps the fifths pure and its
# major thirds are 408 cents, a syntonic comma wide. Quarter-comma meantone narrows each fifth
# by a quarter of the syntonic comma, to {{ f"{so.ratio_to_cents(MEANTONE_FIFTH):.2f}" }} cents,
# so that four of them make a just third; the comma it leaves over goes into one unusable "wolf"
# fifth of {{ f"{8400 - 11 * so.ratio_to_cents(MEANTONE_FIFTH):.1f}" }} cents (usually G♯–E♭),
# which this page's tune never plays. Just intonation on C has pure thirds and fifths in its
# three main chords, but one fifth among its white notes, D–A, is a syntonic comma narrow.

# %% [markdown]
# ## One tune, four tunings
#
# The first line of Twinkle, Twinkle, Little Star, in four parts: the tune in the fourth
# octave, and below it three-note chords in close position (C major, F major, G major) changing
# every two beats. The tune only uses those three chords, which just intonation on C keeps
# exactly just, so this is just intonation at its best; the next section shows where it fails.

# %%
BEAT = 0.45
MELODY = ["C4", "C4", "G4", "G4", "A4", "A4", "G4", None, "F4", "F4", "E4", "E4", "D4", "D4", "C4", None]
CHORDS = {"I": ["C3", "E3", "G3"], "IV": ["F3", "A3", "C4"], "V": ["G2", "B2", "D3"]}
HARMONY = ["I", "I", "IV", "I", "IV", "I", "V", "I"]  # one chord per two beats


def twinkle(tuning, c4=C4, beats=16):
    """Events and Lissajous rows for the first ``beats`` beats of the tune in one tuning."""
    events, chords = [], []
    for beat in range(0, beats, 2):
        chord = [pitch(note, tuning, c4) for note in CHORDS[HARMONY[beat // 2]]]
        events += [(beat * BEAT, 2 * BEAT, f) for f in chord]
    for beat, note in enumerate(MELODY[:beats]):
        if note is None:  # the note before holds for two beats
            continue
        length = 2 if beat + 1 < beats and MELODY[beat + 1] is None else 1
        melody = pitch(note, tuning, c4)
        events.append((beat * BEAT, length * BEAT, melody))
        bass = pitch(CHORDS[HARMONY[beat // 2]][0], tuning, c4)
        chords.append((beat * BEAT, length * BEAT, (bass, melody)))
    return events, figure_notes(chords)


def twinkle_demo(tuning):
    """The tune in one tuning: sound, figure, playhead and Lissajous data."""
    events, notes = twinkle(tuning)
    sound = finish(play(events, len(MELODY) * BEAT + 0.3))
    fig, playhead = show(sound, events)
    lissajous = {
        "notes": notes,
        "window": 0.016,
        "xlabel": "bass",
        "ylabel": "tune",
        "title": "Bass against tune",
        "start": 2.5,
    }
    return sound, fig, playhead, lissajous


# %% [about]
# Equal temperament. Every note sits on the zero line of the lower panel, by definition. The
# fifths beat slowly and the thirds more than ten times faster, and the figure turns through each
# chord; it stands still only on octaves (C over C, F over F), which every tuning here keeps
# pure.

# %% [demo tt2] Twinkle in equal temperament
sound, fig, playhead, lissajous = twinkle_demo("equal temperament")

# %% [about]
# 5-limit just intonation on C. E is 13.7 cents and A 15.6 cents below equal temperament, B
# 11.7 cents below; the chords are smooth, and the figure stands still on every note of the
# tune, since each melody note is a just interval above its bass.

# %% [demo tt3] Twinkle in just intonation
sound, fig, playhead, lissajous = twinkle_demo("just intonation")

# %% [about]
# Pythagorean tuning: every fifth pure, so the figure stands still on C over C, G over C and D
# over G; the major thirds are 408 cents, and the chords beat faster than in equal temperament.

# %% [demo tt4] Twinkle in Pythagorean tuning
sound, fig, playhead, lissajous = twinkle_demo("Pythagorean")

# %% [about]
# Quarter-comma meantone: the thirds are just, as in just intonation, and every fifth is
# 5.4 cents narrower than just (3.4 narrower than equal temperament's), so the fifths beat a
# little faster than in equal temperament while the thirds are calm.

# %% [demo tt5] Twinkle in quarter-comma meantone
sound, fig, playhead, lissajous = twinkle_demo("quarter-comma meantone")

# %% [markdown]
# ## Where just intonation breaks
#
# Just intonation fixed on C has one bad chord among the white notes: D minor, the ii chord,
# whose fifth D–A is a syntonic comma narrow,
# {{ f"{so.ratio_to_cents(TUNINGS['just intonation']['A'] / TUNINGS['just intonation']['D']):.1f}" }}
# cents, and whose minor third D–F is a Pythagorean 294.1 cents instead of 315.6. Four chords,
# C major, D minor, G major, C major, with the top voice rising G, A, B, C, in just intonation
# and then in equal temperament. In the just version the figure stands still on the first,
# third and fourth chords and spins on the second, where the top A is not quite three times
# the bass D; in equal temperament it turns slowly on every chord.

# %%
PROGRESSION = [
    ["C3", "E3", "G3", "G4"],
    ["D3", "F3", "A3", "A4"],
    ["G2", "B2", "D3", "B4"],
    ["C3", "E3", "G3", "C5"],
]
HOLD = 1.6


def progression(tuning):
    chords = [
        (i * HOLD, HOLD - 0.05, [pitch(name, tuning) for name in names])
        for i, names in enumerate(PROGRESSION)
    ]
    events = [(start, duration, f) for start, duration, freqs in chords for f in freqs]
    return events, figure_notes(chords)


# %% [about]
# Just intonation, then equal temperament, four chords each. In the just version listen for
# the second chord, D minor, which beats roughly where its neighbours are smooth.

# %% [demo tt6] C, D minor, G, C, just then equal
just_events, just_notes = progression("just intonation")
equal_events, equal_notes = progression("equal temperament")
gap = len(PROGRESSION) * HOLD + 0.6
events = just_events + [(start + gap, duration, f) for start, duration, f in equal_events]
sound = finish(play(events, 2 * gap))
fig, playhead = show(sound, events)
lissajous = {
    "notes": just_notes + [(t0 + gap, t1 + gap, *rest) for t0, t1, *rest in equal_notes],
    "window": 0.016,
    "xlabel": "bass",
    "ylabel": "top voice",
    "title": "Bass against top voice",
    "start": 2.4,
}

# %% [markdown]
# ## A reference pitch is a choice
#
# All of the above is about the ratios between notes. Which frequency the A above middle C gets
# is a separate choice, the reference pitch. ISO 16 (1975) fixes A4 = 440 Hz; before that,
# reference pitches varied widely between places, periods and kinds of instrument, as Haynes
# (2002) documents at length. A4 = 432 Hz is one more reference: it moves every note down by
# {{ f"{so.ratio_to_cents(440 / 432):.1f}" }} cents, about a third of an equal-tempered semitone,
# and changes nothing else. The health benefits claimed for 432 Hz rest mainly on one small
# pilot study (Calamassi & Pomponi, 2019: 33 listeners, two 20-minute sessions), which reported
# a mean heart rate 4.79 beats per minute lower with the music at 432 Hz (p = 0.05); a word of
# caution, then: that is one pilot result, not evidence that the tuning affects health.

# %% [about]
# The first half of the tune in equal temperament, at A4 = 440 Hz and then at A4 = 432 Hz. In
# the lower panel every note of the second half is 31.8 cents below the zero line; on the
# spectrogram's logarithmic axis the whole picture moves down by the same distance.

# %% [demo tt7] Twinkle at 440 Hz, then at 432 Hz
half = 8
first, _ = twinkle("equal temperament", beats=half)
second, _ = twinkle("equal temperament", c4=so.note_to_freq("C4", a4=432.0), beats=half)
gap = half * BEAT + 0.6
events = first + [(start + gap, duration, f) for start, duration, f in second]
sound = finish(play(events, 2 * gap))
fig, playhead = show(sound, events)

# %% [markdown]
# Keyboard tunings have a long history: roughly in that order, Pythagorean tuning, meantone
# temperaments, the well temperaments, in which every key is usable but each has its own
# shading, and equal temperament, which became the usual tuning of pianos in the nineteenth
# century. This page does not argue that history; Barbour (1951) surveys the tunings with
# their numbers, and Duffin (2007) makes the case for listening to the older ones.

# %% [markdown]
# ## References
#
# - Barbour (1951). *Tuning and Temperament: A Historical Survey*. Michigan State College Press.
# - Calamassi & Pomponi (2019). Music tuned to 440 Hz versus 432 Hz and the health effects: a
#   double-blind cross-over pilot study. *Explore* 15(4), 283–290.
#   [doi:10.1016/j.explore.2019.04.001](https://doi.org/10.1016/j.explore.2019.04.001).
# - Duffin (2007). *How Equal Temperament Ruined Harmony (and Why You Should Care)*. W. W.
#   Norton.
# - Haynes (2002). *A History of Performing Pitch: The Story of "A"*. Scarecrow Press.
# - ISO 16:1975. Acoustics — Standard tuning frequency (standard musical pitch).
#   [ISO](https://www.iso.org/standard/3601.html).
#   [`utils.note_to_freq`](https://github.com/choyun1/sonore/blob/main/src/sonore/core/utils.py#L143)
#   [`utils.ratio_to_cents`](https://github.com/choyun1/sonore/blob/main/src/sonore/core/utils.py#L126)
