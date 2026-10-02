"""Stores librosa's own MFCC output for the MFCC tests (tests/data/librosa_mfcc_reference.npz).

Runs librosa 0.11.0 on the gallery sentence with three settings: librosa's
defaults (Slaney mel, area-normalised triangles, 2048-point windows,
128 bands, 20 coefficients, dB with an 80 dB floor below the loudest cell),
the speech recipe's sizes with HTK mel and height-1 triangles (400-sample
Hann window, 160-sample hop, 512-point FFT, 26 bands, 13 coefficients), and
the same sizes with Slaney mel and 40 bands. For each it stores the mel
power spectrogram and the MFCCs, and for the HTK setting also the first
and second deltas (``librosa.feature.delta`` with width 5 and 9). The tests compare sonore
against these numbers, so they need no librosa. librosa is used only here.

    pip install librosa            # development only
    python tools/make_mfcc_fixtures.py
"""

from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SPEECH = ROOT / "docs" / "speech" / "bdl_arctic_a0131.flac"
OUT = ROOT / "tests" / "data" / "librosa_mfcc_reference.npz"

SETTINGS = {
    "defaults": dict(
        n_fft=2048, hop_length=512, win_length=2048, n_mels=128, n_mfcc=20, htk=False, norm="slaney"
    ),
    "speech_htk": dict(n_fft=512, hop_length=160, win_length=400, n_mels=26, n_mfcc=13, htk=True, norm=None),
    "speech_slaney": dict(
        n_fft=512, hop_length=160, win_length=400, n_mels=40, n_mfcc=13, htk=False, norm="slaney"
    ),
}


def main():
    speech, fs = sf.read(SPEECH)
    arrays = {"librosa_version": np.array(librosa.__version__)}
    for name, setting in SETTINGS.items():
        mel_power = librosa.feature.melspectrogram(
            y=speech,
            sr=fs,
            n_fft=setting["n_fft"],
            hop_length=setting["hop_length"],
            win_length=setting["win_length"],
            window="hann",
            n_mels=setting["n_mels"],
            htk=setting["htk"],
            norm=setting["norm"],
        )
        coefficients = librosa.feature.mfcc(S=librosa.power_to_db(mel_power), n_mfcc=setting["n_mfcc"])
        arrays[f"{name}_mel_power"] = mel_power
        arrays[f"{name}_mfcc"] = coefficients
        if name != "speech_htk":
            continue
        for width in (5, 9):
            arrays[f"{name}_delta{width}"] = librosa.feature.delta(coefficients, width=width)
            arrays[f"{name}_delta{width}_order2"] = librosa.feature.delta(coefficients, width=width, order=2)
    OUT.parent.mkdir(exist_ok=True)
    np.savez_compressed(OUT, **arrays)
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.0f} KiB)")


if __name__ == "__main__":
    main()
