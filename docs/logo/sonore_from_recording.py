"""From a recording of "sonore" to a formant-synthesizer copy of it, with nothing set by hand.

The example "sonore says its name" on the Formant synthesis page (which the logo is drawn from)
uses tracks read by hand off the table this script prints. This script makes the automatic copy
instead, so the two can be compared.

The recording is the word as pronounced on https://www.frenchdictionary.com/translate/sonore,
saved as an audio file. It is not in the repository. Run from the repository root with its path:

    python docs/logo/sonore_from_recording.py path/to/sonore.wav

The script goes through four steps:

1. Analysis: the recording at 16 kHz, cut to the word (where it is within 40 dB of its loudest).
2. Measurement: every 5 ms, the level, the share of the power above 3.5 kHz (high for the /s/),
   whether it is voiced and its F0 (so.f0_track), and F1 to F3 (so.formant_track, with a
   five-window median to take out single-window jumps). It prints these every 20 ms as a table.
3. Synthesis: so.klatt_synthesize gets those tracks directly. Where the word is voiced, the
   voicing level follows the recording's level. Where it is unvoiced, the noise follows that
   level too: frication through the high parallel formants where most of the power is above
   3.5 kHz (the /s/), and aspiration through the tracked formants elsewhere (the /ʁ/). The copy
   is made twice, the second time with each setting moved by how far the first copy's level
   missed the recording's, since the three sources come out at different levels.
4. Comparison: the two level profiles side by side, both spectrograms with the formant tracks
   drawn on them (sonore_from_recording.png), and the recording followed by the copy
   (sonore_from_recording.wav), both written next to the recording, since they contain it.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter

import sonore as so

FS = 16000
HOP = 0.005  # seconds between measurements, as in so.f0_track and so.formant_track
WINDOW = 0.02  # seconds over which the level and the high-frequency share are measured


def levels_db(samples, times):
    """Level in dB of WINDOW seconds of `samples` centered on each of `times`."""
    window = round(WINDOW * FS)
    starts = np.clip(np.round(times * FS).astype(int) - window // 2, 0, len(samples) - window)
    return np.array([10 * np.log10(np.mean(samples[s : s + window] ** 2) + 1e-20) for s in starts])


def high_share_db(samples, times, cutoff=3500.0):
    """Share of the power above `cutoff`, in dB, in WINDOW seconds centered on each of `times`."""
    window = round(WINDOW * FS)
    starts = np.clip(np.round(times * FS).astype(int) - window // 2, 0, len(samples) - window)
    above = np.fft.rfftfreq(window, 1 / FS) > cutoff
    shares = []
    for s in starts:
        power = np.abs(np.fft.rfft(samples[s : s + window] * np.hanning(window))) ** 2 + 1e-20
        shares.append(10 * np.log10(power[above].sum() / power.sum()))
    return np.array(shares)


def main(path):
    # 1. Analysis: the recording at 16 kHz, cut to the word.
    samples = so.load(path).resample(FS).data[:, 0]
    grid = np.arange(0, len(samples) / FS, HOP)
    level = levels_db(samples, grid)
    loud = grid[level > level.max() - 40]
    first, last = round((loud[0] - WINDOW) * FS), round((loud[-1] + WINDOW) * FS)
    word = so.Sound(samples[first:last, None], FS)
    print(f"The word runs from {first / FS:.2f} s to {last / FS:.2f} s of the recording.")

    # 2. Measurement: level, high-frequency share, F0 and F1 to F3 every 5 ms.
    x = word.data[:, 0]
    pitch = so.f0_track(word, f_lo=60, f_hi=400)
    median_f0 = np.median(pitch.f0[0][pitch.voiced[0]])
    ceiling = 5500 if median_f0 > 160 else 5000  # Praat's ceilings for female and male voices
    formants = so.formant_track(word, ceiling=ceiling)
    print(f"Median F0 {median_f0:.0f} Hz, so a formant ceiling of {ceiling} Hz.")
    times = formants.t[(formants.t >= 0) & (formants.t <= word.duration)]
    tracks = np.nan_to_num(formants.frequencies[0], nan=1000.0)
    tracks = median_filter(np.array([np.interp(times, formants.t, f) for f in tracks]), size=(1, 5))
    voiced = np.interp(times, pitch.t, pitch.voiced[0].astype(float)) > 0.5
    f0 = np.interp(times, pitch.t, np.where(pitch.voiced[0], pitch.f0[0], 0.0))
    level = levels_db(x, times)
    level -= level.max()
    high = high_share_db(x, times)
    print("   t   level  >3.5 kHz  voiced    F0     F1     F2     F3")
    for k in range(0, len(times), round(0.02 / HOP)):
        f0_text = f"{f0[k]:5.0f}" if voiced[k] else "    -"
        f1, f2, f3 = tracks[:, k]
        yes_no = "yes" if voiced[k] else " no"
        columns = f"{yes_no}  {f0_text} {f1:6.0f} {f2:6.0f} {f3:6.0f}"
        print(f"{times[k]:5.2f} {level[k]:6.1f} dB {high[k]:6.1f} dB  {columns}")

    # 3. Synthesis: the measured tracks, every 5 ms, straight into the synthesizer. Levels are
    # in dB with 60 as the loudest; nothing more than 40 dB down is synthesized.
    audible = level > -40
    fricative = ~voiced & audible & (high > -6)
    aspirated = ~voiced & audible & ~fricative

    def synthesize(level):
        return so.klatt_synthesize(
            word.duration,
            FS,
            F0=(times, f0),
            AV=(times, np.where(voiced, 60 + level, 0)),
            AF=(times, np.where(fricative, 60 + level, 0)),
            AH=(times, np.where(aspirated, 60 + level, 0)),
            F1=(times, tracks[0]),
            F2=(times, tracks[1]),
            F3=(times, tracks[2]),
            F4=4400,
            F5=5200,
            F6=6500,
            B6=1500,
            A4=34,
            A5=50,
            A6=60,
            rng=1,
        )

    # The voicing, the frication and the aspiration come out at different levels for the same
    # setting, so the copy is made twice: the second time with each window's setting moved by
    # how far the first copy's level missed the recording's there.
    first_copy = synthesize(level)
    first_level = levels_db(first_copy.data[:, 0], times)
    miss = level - (first_level - first_level.max())
    copy = synthesize(level + np.where(audible, miss, 0))

    # 4. Comparison: level every 20 ms, spectrograms, and the two sounds one after the other.
    y = copy.data[:, 0]
    every = times[:: round(0.02 / HOP)]
    rec_level, copy_level = levels_db(x, every), levels_db(y, every)
    rec_level, copy_level = rec_level - rec_level.max(), copy_level - copy_level.max()
    print("   t   recording  copy (dB re loudest)")
    for t, rec_db, copy_db in zip(every, rec_level, copy_level, strict=True):
        print(f"{t:5.2f} {rec_db:9.1f} {copy_db:6.1f}")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, layout="constrained")
    for ax, (title, sound) in zip(axes, {"Recording": word, "Automatic copy": copy}.items(), strict=True):
        so.STFT(sound, win_dur=0.006, hop_dur=0.001).plot(ax, db_range=60, colorbar=False, fmax=8000)
        for track in tracks:
            ax.plot(times, np.where(voiced | aspirated, track, np.nan) / 1000, "c-", lw=0.8)  # kHz
        ax.set_title(title)
    out = Path(path).parent
    fig.savefig(out / "sonore_from_recording.png", dpi=100)
    gap = so.silence(0.5, FS)
    both = so.concat([word.normalize(rms=0.1), gap, copy.normalize(rms=0.1)])
    both.normalize(peak=0.95).save(out / "sonore_from_recording.wav")


if __name__ == "__main__":
    main(sys.argv[1])
