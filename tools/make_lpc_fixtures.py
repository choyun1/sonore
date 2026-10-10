"""Stores Praat's formants for the formant tracking tests (tests/data/praat_formants_reference.npz).

Runs Praat's ``To Formant (burg)`` (parselmouth) on 1 s of each gallery
sentence (0.5 to 1.5 s, stored in the file as 16-bit samples, so the tests
need no docs folder): time step 5 ms, 5 formants, ceiling 5000 Hz for the
male talker (bdl) and 5500 Hz for the female one (slt), 25 ms window. It
stores F1 to F3 at sonore's window times (multiples of 5 ms from the start of
the excerpt) and which of those times the Harvest track
(docs/speech/*_f0.csv) calls voiced. The tests compare
``so.formant_track`` with these numbers, so they need no Praat. parselmouth
is used only here.

    pip install praat-parselmouth   # development only
    python tools/make_lpc_fixtures.py
"""

from pathlib import Path

import numpy as np
import parselmouth
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SPEECH = ROOT / "docs" / "speech"
OUT = ROOT / "tests" / "data" / "praat_formants_reference.npz"
START, STOP = 0.5, 1.5
CEILINGS = {"bdl": 5000, "slt": 5500}


def main():
    stored = {"parselmouth_version": parselmouth.__version__, "praat_version": parselmouth.PRAAT_VERSION}
    for speaker, ceiling in CEILINGS.items():
        signal, fs = sf.read(SPEECH / f"{speaker}_arctic_a0131.flac", dtype="int16")
        excerpt = signal[round(START * fs) : round(STOP * fs)]
        formants = parselmouth.Sound(excerpt / 32768.0, fs).to_formant_burg(
            time_step=0.005, max_number_of_formants=5, maximum_formant=ceiling, window_length=0.025
        )
        times = np.arange(0, STOP - START + 1e-9, 0.005)
        praat = np.array(
            [[formants.get_value_at_time(number, time) for number in (1, 2, 3)] for time in times]
        )
        harvest = np.loadtxt(SPEECH / f"{speaker}_arctic_a0131_f0.csv", delimiter=",", skiprows=2)
        voiced = np.interp(times + START, harvest[:, 0], harvest[:, 1]) > 0
        stored |= {
            f"{speaker}_samples_int16": excerpt,
            f"{speaker}_fs": fs,
            f"{speaker}_ceiling": ceiling,
            f"{speaker}_times": times,
            f"{speaker}_formants": praat.T,
            f"{speaker}_voiced": voiced,
        }
    np.savez_compressed(OUT, **stored)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
