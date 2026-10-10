"""From a recording of "sonore" to the formant-synthesizer version on the Formant synthesis page
(the example "sonore says its name", which the logo is drawn from).

The recording is the word as pronounced on https://www.frenchdictionary.com/translate/sonore,
saved as an audio file. It is not in the repository. Run from the repository root with its path:

    python docs/logo/sonore_from_recording.py path/to/sonore.wav

The script goes through four steps:

1. Analysis: the recording at 16 kHz, cut to the word (where it is within 40 dB of its loudest).
2. Measurement: every 20 ms, the level, the share of the power above 3.5 kHz (high for the /s/),
   whether it is voiced and its F0 (so.f0_track), and F1 to F3 (so.formant_track). It prints
   these as a table.
3. Synthesis: the tracks for so.klatt_synthesize, set by hand from that table. These numbers are
   the ones in the gallery example. They were read off the table, and the levels were adjusted
   until the synthesis followed the recording's level within a few dB.
4. Comparison: the two level profiles side by side, both spectrograms with the tracks drawn on
   them (sonore_from_recording.png), and the recording followed by the synthesis
   (sonore_from_recording.wav), both written next to the recording, since they contain it.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sonore as so

FS = 16000

# 3. The synthesis, as in the gallery example. Times are from the start of the /s/.
DUR = 0.62
times = [0.15, 0.21, 0.23, 0.27, 0.29, 0.35, 0.42, 0.46, 0.49, 0.58]  # o, n, o, r
TRACKS = dict(
    F1=(times, [530, 470, 380, 380, 600, 590, 590, 600, 750, 750]),
    F2=(times, [1750, 1730, 1750, 1750, 1800, 1450, 1250, 1200, 1250, 1250]),
    F3=(times, [3000, 2950, 2800, 2800, 2950, 2900, 2970, 3050, 2900, 2900]),
)
PARAMS = dict(
    F0=([0.14, 0.18, 0.22, 0.26, 0.30, 0.34, 0.38, 0.47], [266, 250, 232, 212, 205, 197, 192, 189]),
    AV=(
        [0.135, 0.14, 0.16, 0.21, 0.23, 0.27, 0.29, 0.43, 0.46, 0.48],
        [0, 56, 60, 56, 50, 50, 55, 54, 47, 0],
    ),
    AF=([0.0, 0.02, 0.125, 0.145], [0, 48, 48, 0]),  # the /s/, through the parallel formants
    AH=([0.45, 0.48, 0.55, 0.60], [0, 34, 32, 0]),  # the voiceless /ʁ/, through the cascade
    F4=4400,
    F5=5200,
    F6=6500,
    B6=1500,
    SS=3,
    RD=1.3,
    A4=34,
    A5=50,
    A6=60,
    FNZ=([0.21, 0.23, 0.27, 0.29], [250, 450, 450, 250]),  # the nasal zero opens for the /n/
)


def level_db(samples, starts, window):
    """Level in dB of `window` samples from each start."""
    return np.array([10 * np.log10(np.mean(samples[s : s + window] ** 2) + 1e-20) for s in starts])


def main(path):
    # 1. Analysis: the recording at 16 kHz, cut to the word.
    recording = so.load(path).resample(FS)
    samples = recording.data[:, 0]
    window, hop = round(0.02 * FS), round(0.005 * FS)
    starts = np.arange(0, len(samples) - window, hop)
    level = level_db(samples, starts, window)
    loud = np.flatnonzero(level > level.max() - 40)
    first, last = starts[loud[0]], starts[loud[-1]] + window
    word = so.Sound(samples[first:last, None], FS)
    print(f"The word runs from {first / FS:.2f} s to {last / FS:.2f} s of the recording.")

    # 2. Measurement: level, high-frequency share, F0 and F1 to F3 every 20 ms.
    pitch = so.f0_track(word, f_lo=60, f_hi=400)
    median_f0 = np.median(pitch.f0[0][pitch.voiced[0]])
    ceiling = 5500 if median_f0 > 160 else 5000  # Praat's ceilings for female and male voices
    formants = so.formant_track(word, ceiling=ceiling)
    print(f"Median F0 {median_f0:.0f} Hz, so a formant ceiling of {ceiling} Hz.")
    x = word.data[:, 0]
    word_level = level_db(x, np.arange(0, len(x) - window, hop), window)
    peak = word_level.max()
    print("   t   level  >3.5 kHz  voiced    F0     F1     F2     F3")
    for t in np.arange(0, word.duration - 0.02, 0.02):
        segment = x[round(t * FS) : round(t * FS) + window]
        power = np.abs(np.fft.rfft(segment * np.hanning(len(segment)))) ** 2
        high = 10 * np.log10(power[np.fft.rfftfreq(len(segment), 1 / FS) > 3500].sum() / power.sum())
        i, j = np.argmin(abs(pitch.t - t)), np.argmin(abs(formants.t - t))
        f1, f2, f3 = formants.frequencies[0][:, j]
        voiced = bool(pitch.voiced[0][i])
        f0 = f"{pitch.f0[0][i]:5.0f}" if voiced else "    -"
        level_here = word_level[min(round(t / 0.005), len(word_level) - 1)] - peak
        yes_no = "yes" if voiced else " no"
        print(f"{t:5.2f} {level_here:6.1f} dB {high:6.1f} dB  {yes_no}  {f0} {f1:6.0f} {f2:6.0f} {f3:6.0f}")

    # 3. Synthesis from the tracks set by hand from that table.
    synthesis = so.klatt_synthesize(DUR, FS, rng=1, **PARAMS, **TRACKS)

    # 4. Comparison: level every 20 ms, spectrograms, and the two sounds one after the other.
    y = synthesis.data[:, 0]
    n = min(len(x), len(y))
    compare_starts = np.arange(0, n - window, 2 * hop * 2)
    rec_level, syn_level = level_db(x, compare_starts, window), level_db(y, compare_starts, window)
    print("   t   recording  synthesis (dB re loudest)")
    for t, a, b in zip(
        compare_starts / FS, rec_level - rec_level.max(), syn_level - syn_level.max(), strict=True
    ):
        print(f"{t:5.2f} {a:9.1f} {b:10.1f}")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, layout="constrained")
    for ax, (title, sound) in zip(
        axes, {"Recording": word, "Formant synthesis": synthesis}.items(), strict=True
    ):
        so.STFT(sound, win_dur=0.006, hop_dur=0.001).plot(ax, db_range=60, colorbar=False, fmax=8000)
        for name in ("F1", "F2", "F3"):
            ax.plot(TRACKS[name][0], np.array(TRACKS[name][1]) / 1000, "c--", lw=1)  # the axis is in kHz
        ax.set_title(title)
    fig.savefig(Path(path).parent / "sonore_from_recording.png", dpi=100)
    gap = so.silence(0.5, FS)
    both = so.concat([word.normalize(rms=0.1), gap, synthesis.normalize(rms=0.1)])
    both.normalize(peak=0.95).save(Path(path).parent / "sonore_from_recording.wav")


if __name__ == "__main__":
    main(sys.argv[1])
