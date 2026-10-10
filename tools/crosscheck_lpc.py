"""Cross-check the LPC formants of the gallery sentences against Praat.

Praat's ``To Formant (burg)`` (through parselmouth) is a different method:
Burg's algorithm rather than the autocorrelation method, on a Gaussian
window, after resampling to twice the formant ceiling (5000 Hz for the male
talker, 5500 Hz for the female, Praat's advice) with 5 formants. Neither is
the truth for a recording, so this measures agreement, not error. The LPC
side is the formula-level code of ``tools/check_lpc_claims.py`` (order 18,
25 ms Hamming, pre-emphasis 0.97, the k-th narrow root as Fk), at the times
the Harvest track calls voiced.

Praat is a development-time dependency only:

    pip install praat-parselmouth
    python tools/crosscheck_lpc.py
"""

import numpy as np
import parselmouth
import soundfile as sf
from check_lpc_claims import FS, SPEECH, formants_from_roots, hamming, lpc, preemphasize

CEILINGS = {"bdl": 5000, "slt": 5500}


def report(text, value):
    print(f"{text:<78s} {value:.4g}")


def differences_from_praat(speaker, order):
    """|LPC - Praat| for F1-F3 [Hz] at the voiced times where both give three formants."""
    length = round(0.025 * FS)
    window = hamming(length)
    signal, fs = sf.read(SPEECH / f"{speaker}_arctic_a0131.flac")
    assert fs == FS
    praat = parselmouth.Sound(signal, fs).to_formant_burg(
        time_step=0.005, max_number_of_formants=5, maximum_formant=CEILINGS[speaker], window_length=0.025
    )
    emphasized = preemphasize(signal)
    track = np.loadtxt(SPEECH / f"{speaker}_arctic_a0131_f0.csv", delimiter=",", skiprows=2)
    differences = []
    for time, f0 in track:
        start = round(time * FS) - length // 2
        if f0 <= 0 or start < 0 or start + length > len(signal):
            continue
        freqs, _ = formants_from_roots(lpc(emphasized[start : start + length] * window, order)[0])
        reference = [praat.get_value_at_time(number, time) for number in (1, 2, 3)]
        if len(freqs) < 3 or np.isnan(reference).any():
            continue
        differences.append(freqs[:3] - reference)
    return np.abs(differences)


def main():
    for speaker in CEILINGS:
        differences = differences_from_praat(speaker, 18)
        report(f"{speaker}: voiced times where both give three formants", len(differences))
        for index in range(3):
            report(f"{speaker} F{index + 1}: median |LPC - Praat| [Hz]", np.median(differences[:, index]))
        for index, typical in enumerate((500, 1500, 2500)):
            report(
                f"{speaker} F{index + 1}: fraction of times within 10% of {typical} Hz",
                np.mean(differences[:, index] < 0.1 * typical),
            )
    # Labelling the k-th narrow root as Fk is fragile: the order moves the agreement a lot.
    for speaker in CEILINGS:
        for order in (14, 16, 18, 20):
            differences = differences_from_praat(speaker, order)
            report(f"{speaker}, order {order}: median |LPC - Praat| of F2 [Hz]", np.median(differences[:, 1]))


if __name__ == "__main__":
    main()
