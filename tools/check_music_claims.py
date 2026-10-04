"""Numerical checks for the claims in docs/design/music/music-pages.md (C1-C10).

Like the other design checkers, this is independent of sonore: plain Python
and NumPy, every formula written out. Each line prints the claim number and
the number that supports it.

    python tools/check_music_claims.py
"""

import math

import numpy as np

FS = 44100  # sample rate for the Lissajous check [Hz]


def report(claim, text, value):
    print(f"{claim:4s} {text:<74s} {value:.4f}")


def cents(ratio):
    return 1200 * math.log2(ratio)


def equal(semitones):
    return 2 ** (semitones / 12)


# C1: a reference of A4 = 432 Hz against 440 Hz
report("C1", "A4 = 432 Hz below 440 Hz [cents]", cents(440 / 432))
report("C1", "same, as a fraction of an equal-tempered semitone", cents(440 / 432) / 100)

# C2: just, Pythagorean and equal-tempered intervals
for name, ratio, semitones in [
    ("fifth 3:2", 3 / 2, 7),
    ("fourth 4:3", 4 / 3, 5),
    ("major third 5:4", 5 / 4, 4),
    ("minor third 6:5", 6 / 5, 3),
    ("major sixth 5:3", 5 / 3, 9),
]:
    report("C2", f"{name}: just [cents]", cents(ratio))
    report("C2", f"{name}: equal minus just [cents]", 100 * semitones - cents(ratio))
report("C2", "Pythagorean major third 81:64 [cents]", cents(81 / 64))

# C3: the commas
report("C3", "Pythagorean comma, 12 fifths over 7 octaves (3^12/2^19) [cents]", cents(3**12 / 2**19))
report("C3", "syntonic comma 81:80 [cents]", cents(81 / 80))
report("C3", "quarter-comma meantone fifth [cents]", cents(3 / 2) - cents(81 / 80) / 4)
meantone_fifth = cents(3 / 2) - cents(81 / 80) / 4
report("C3", "meantone wolf fifth, 7 octaves less 11 meantone fifths [cents]", 8400 - 11 * meantone_fifth)

# C4: beats between the nearest coinciding partials of tempered intervals over A3 = 220 Hz
A3 = 220.0
for name, semitones, lower_n, upper_n in [
    ("fifth (3rd of A3 vs 2nd of E4)", 7, 3, 2),
    ("fourth (4th of A3 vs 3rd of D4)", 5, 4, 3),
    ("major third (5th of A3 vs 4th of C#4)", 4, 5, 4),
    ("minor third (6th of A3 vs 5th of C4)", 3, 6, 5),
]:
    upper = A3 * equal(semitones)
    report("C4", f"equal {name} beat [Hz]", abs(upper_n * upper - lower_n * A3))

# C5: a Lissajous figure of a near-3:2 pair repeats its shape at |2 f_y - 3 f_x|.
# Sample x = cos(2 pi f_x t), y = cos(2 pi f_y t), and for each 2/f_x-long stretch find the
# shift of y (as a phase of the 3:2 figure) that best matches the just-tuned figure.
f_x = A3
f_y = A3 * equal(7)
duration = 4.0
t = np.arange(int(duration * FS)) / FS
x = np.cos(2 * np.pi * f_x * t)
y = np.cos(2 * np.pi * f_y * t)
span = int(round(2 / f_x * FS))  # one period of the just 3:2 figure
starts = np.arange(0, len(t) - span, span)
trial_phases = np.linspace(0, 2 * np.pi, 720, endpoint=False)
best = []
for start in starts:
    local_t = t[start : start + span]
    yy = y[start : start + span]
    candidates = np.cos(2 * np.pi * 1.5 * f_x * local_t[None, :] + trial_phases[:, None])
    best.append(trial_phases[np.argmax(candidates @ yy)])
unwrapped = np.unwrap(np.array(best))
rate = abs(np.polyfit(t[starts], unwrapped, 1)[0]) / (2 * np.pi)
# the figure's shape depends on 2*theta_y - 3*theta_x; y's phase moves 1/2 as fast as that
report("C5", "Lissajous A3 + equal E4: y phase drift [cycles/s]", rate)
report("C5", "predicted |2 f_y - 3 f_x| / 2 [cycles/s]", abs(2 * f_y - 3 * f_x) / 2)
report("C5", "time to pass through every shape, 1 / |2 f_y - 3 f_x| [s]", 1 / abs(2 * f_y - 3 * f_x))

# C6: pipe footage is nominal. An open pipe sounds near c / (2 L); 8 ft is low C, C2 = 65.41 Hz.
c_air = 343.0  # m/s at about 20 degrees C
eight_feet = 8 * 0.3048
report("C6", "C2 in equal temperament from A4 = 440 [Hz]", 440 * equal(-33))
report("C6", "c / (2 * 8 ft), no end correction [Hz]", c_air / (2 * eight_feet))
report("C6", "length an ideal open pipe needs for C2 [ft]", c_air / (2 * 440 * equal(-33)) / 0.3048)

# C7: footages and the harmonic each stop reinforces: harmonic n of an 8' stop is 8/n feet
for footage, n in [(16, 0.5), (8, 1), (4, 2), (8 / 3, 3), (2, 4), (8 / 5, 5), (4 / 3, 6), (8 / 7, 7), (1, 8)]:
    report("C7", f"{footage:6.4f} ft = 8 / harmonic {n:g}", 8 / footage)

# C8: mutations tuned pure against tempered (as on a tonewheel organ): deviation and beat rate
# against the 8' stop's own partial, played on middle C (C4 = 261.63 Hz)
C4 = 440 * equal(-9)
MUTATIONS = [
    ("twelfth 2 2/3'", 3, 19),
    ("tierce 1 3/5'", 5, 28),
    ("larigot 1 1/3'", 6, 31),
    ("septieme 1 1/7'", 7, 34),
]
for name, n, semitones in MUTATIONS:
    tempered = C4 * equal(semitones)
    report("C8", f"{name}: tempered minus pure [cents]", cents(tempered / (n * C4)))
    report("C8", f"{name}: beat with the 8' stop's harmonic {n} on C4 [Hz]", abs(tempered - n * C4))

# C9: a celeste rank tuned a few cents sharp beats with the unison rank at f * (2^(c/1200) - 1)
for detune in (3, 6, 10):
    report("C9", f"celeste {detune} cents sharp, beat on C4 [Hz]", C4 * (2 ** (detune / 1200) - 1))

# C10: 5-limit just intonation fixed on C (D = 9/8, A = 5/3) keeps I, IV and V just, so a
# tune that uses only those chords (Twinkle, Twinkle) never meets a bad interval; the fifth
# D-A of the ii chord is a syntonic comma narrow, and its minor third D-F is Pythagorean
just_c = {"C": 1, "D": 9 / 8, "E": 5 / 4, "F": 4 / 3, "G": 3 / 2, "A": 5 / 3, "B": 15 / 8}
for chord, (root, third, fifth) in {"I": "CEG", "IV": "FAC", "V": "GBD", "ii": "DFA"}.items():
    for name, note in (("third", third), ("fifth", fifth)):
        upper = just_c[note] * (2 if just_c[note] < just_c[root] else 1)
        report("C10", f"just-on-C {chord} chord: {name} {root}-{note} [cents]", cents(upper / just_c[root]))
