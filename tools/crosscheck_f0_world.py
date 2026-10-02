"""WORLD's own F0 estimators on the test set of docs/design/f0.md.

Runs Harvest and DIO (followed by StoneMask) through pyworld on the
synthetic cases of tools/check_f0_claims.py, on white noise alone, and on
the two held-out `rms` sentences in docs/speech, where the checker's
prototype tracker is compared with Harvest. pyworld is used only here, as a
comparison oracle; it is not a sonore dependency.

    pip install pyworld            # development only
    python tools/crosscheck_f0_world.py
"""

from pathlib import Path

import numpy as np
import pyworld
import soundfile as sf
from check_f0_claims import (
    CONTOURS,
    F_HI,
    F_LO,
    HOP,
    HP300,
    NOISE,
    SPEECH,
    VIBRATO,
    cepstral_track,
    errors,
    report,
    track,
    vowel,
    with_noise,
)
from scipy.signal import sosfiltfilt

FS = 16000


def world(x, method):
    x = np.ascontiguousarray(x, dtype=float)
    kw = dict(f0_floor=F_LO, f0_ceil=F_HI, frame_period=HOP * 1000)
    if method == "Harvest":
        f0, t = pyworld.harvest(x, FS, **kw)
    else:
        f0, t = pyworld.dio(x, FS, **kw)
        f0 = pyworld.stonemask(x, f0, t, FS)
    return t, f0


def main():
    print(f"pyworld {pyworld.__version__}")
    cases = {k: (v, None) for k, v in CONTOURS.items()}
    for snr in (20, 10, 0):
        cases[f"vibrato, white noise {snr} dB SNR"] = (VIBRATO, snr)
    cases["vibrato, high-passed at 300 Hz"] = (VIBRATO, "hp")
    for label, (f0, mod) in cases.items():
        x = vowel(f0)
        if mod == "hp":
            x = sosfiltfilt(HP300, x)
        elif mod is not None:
            x = with_noise(x, mod)
        for method in ("Harvest", "DIO + StoneMask"):
            g, med, mx = errors(*world(x, method), f0)
            report("W", f"{label} | {method}: gross error rate", g)
            report("W", f"{label} | {method}: median fine error, %", med)
    for method in ("Harvest", "DIO + StoneMask"):
        t, f0 = world(NOISE, method)
        report(
            "W", f"one second of white noise, {method}: time windows called voiced, fraction", np.mean(f0 > 0)
        )

    # Held-out sentences: the prototype's settings were chosen on bdl.
    for path in sorted(Path(SPEECH).glob("rms_*.flac")):
        x, fs = sf.read(path)
        assert fs == FS
        t_h, h = world(x, "Harvest")
        for name, (t, est) in [("tracker", track(x)), ("cepstral baseline", cepstral_track(x))]:
            hh = np.interp(t, t_h, h)
            hv = np.interp(t, t_h, (h > 0).astype(float)) > 0.5
            v = est > 0
            both = v & hv
            report("W", f"{path.stem}, {name}: share of Harvest-voiced time windows voiced", np.mean(v[hv]))
            report("W", f"{path.stem}, {name}: share of Harvest-unvoiced voiced", np.mean(v[~hv]))
            report(
                "W",
                f"{path.stem}, {name}: both voiced, within 5% of Harvest, fraction",
                np.mean(np.abs(est[both] / hh[both] - 1) < 0.05),
            )
            if name != "tracker":
                continue
            # Where the Harvest-voiced time windows the tracker leaves out lie.
            miss, hit = hv & ~v, hv & v
            L = 400
            xp = np.concatenate([np.zeros(L), x, np.zeros(L)])
            idx = np.round(t * FS).astype(int) + L // 2
            level = 10 * np.log10(np.array([np.mean(xp[i : i + L] ** 2) for i in idx]) + 1e-20)
            edges = np.flatnonzero(np.diff(np.r_[0, miss.astype(int), 0]))
            report(
                "W",
                f"{path.stem}: longest run of those time windows, ms",
                np.max(edges[1::2] - edges[::2]) * HOP * 1000,
            )
            report(
                "W",
                f"{path.stem}: their median 25 ms level re the voiced time windows', dB",
                np.median(level[miss]) - np.median(level[hit]),
            )
            report("W", f"{path.stem}: their median Harvest F0, Hz", np.median(hh[miss]))
            report(
                "W", f"{path.stem}: median Harvest F0 of the time windows it voices, Hz", np.median(hh[hit])
            )


if __name__ == "__main__":
    main()
