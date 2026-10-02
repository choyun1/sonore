"""Stores WORLD's own output for the vocoder tests (tests/data/world_reference.npz).

Runs pyworld (WORLD's C++ code, version 0.3.5, the version the ports were
checked against) on two short sounds: a synthetic vowel with vibrato and breath noise,
and 0.3 s of the gallery sentence (0.6 to 0.9 s) with its stored Harvest track. It
stores every 8th frequency bin of CheapTrick's envelope (with WORLD's q1 and
the paper's), D4C's aperiodicity, and WORLD's synthesis from WORLD's own
envelope and aperiodicity. The tests compare sonore against these numbers, so
they need no pyworld. pyworld is used only here.

    pip install pyworld            # development only
    python tools/make_world_fixtures.py
"""

from pathlib import Path

import numpy as np
import pyworld
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "data" / "world_reference.npz"
FS = 16000
HOP_MS = 5.0
BIN_STEP = 8


def breathy_vowel(duration=0.3, seed=7):
    """Harmonics of a 120 Hz F0 with a 5.5 Hz, 3% vibrato, falling 12 dB per
    octave, plus white noise 25 dB down."""
    rng = np.random.default_rng(seed)
    sample_times = np.arange(int(duration * FS)) / FS
    f0_per_sample = 120 * (1 + 0.03 * np.sin(2 * np.pi * 5.5 * sample_times))
    phase = np.cumsum(2 * np.pi * f0_per_sample / FS)
    harmonic_numbers = np.arange(1, 60)
    amplitudes = 1.0 / harmonic_numbers**2
    sound = np.zeros_like(sample_times)
    for number, amplitude in zip(harmonic_numbers, amplitudes, strict=True):
        below_nyquist = number * f0_per_sample < 0.45 * FS
        sound += below_nyquist * amplitude * np.cos(number * phase)
    sound += rng.standard_normal(len(sound)) * np.std(sound) * 10 ** (-25 / 20)
    window_times = np.arange(0, duration, HOP_MS / 1000)
    window_f0 = 120 * (1 + 0.03 * np.sin(2 * np.pi * 5.5 * window_times))
    return 0.3 * sound / np.abs(sound).max(), window_times, window_f0


def sentence_excerpt(start=0.6, stop=0.9):
    """0.3 s of the gallery sentence and its Harvest track, both
    shifted to start at 0 so they can be resynthesized."""
    sound, fs = sf.read(ROOT / "docs" / "speech" / "bdl_arctic_a0131.flac")
    if fs != FS:
        raise SystemExit(f"expected {FS} Hz, got {fs}")
    track = np.loadtxt(ROOT / "docs" / "speech" / "bdl_arctic_a0131_f0.csv", delimiter=",", skiprows=2)
    rows = (track[:, 0] >= start - 1e-9) & (track[:, 0] < stop - 1e-9)
    excerpt = sound[int(round(start * FS)) : int(round(stop * FS))]
    return excerpt, track[rows, 0] - start, track[rows, 1]


def world_outputs(sound, window_times, window_f0):
    sound, window_times, window_f0 = (
        np.ascontiguousarray(values, dtype=float) for values in (sound, window_times, window_f0)
    )
    envelope = pyworld.cheaptrick(sound, window_f0, window_times, FS)
    envelope_paper = pyworld.cheaptrick(sound, window_f0, window_times, FS, q1=-0.09)
    aperiodicity = pyworld.d4c(sound, window_f0, window_times, FS)
    synthesized = pyworld.synthesize(window_f0, envelope, aperiodicity, FS, HOP_MS)
    return {
        "sound": sound,
        "t": window_times,
        "f0": window_f0,
        "envelope": envelope[:, ::BIN_STEP],
        "envelope_paper_q1": envelope_paper[:, ::BIN_STEP],
        "aperiodicity": aperiodicity[:, ::BIN_STEP],
        "synthesized": synthesized,
    }


def main():
    arrays = {"pyworld_version": np.array(pyworld.__version__), "bin_step": np.array(BIN_STEP)}
    for name, (sound, window_times, window_f0) in {
        "vowel": breathy_vowel(),
        "sentence": sentence_excerpt(),
    }.items():
        for key, value in world_outputs(sound, window_times, window_f0).items():
            arrays[f"{name}_{key}"] = value
    OUT.parent.mkdir(exist_ok=True)
    np.savez_compressed(OUT, **arrays)
    print(f"wrote {OUT.relative_to(ROOT)}, {OUT.stat().st_size / 1024:.0f} kB, pyworld {pyworld.__version__}")


if __name__ == "__main__":
    main()
