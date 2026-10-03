"""WORLD's own CheapTrick, D4C and synthesis against the checker's ports
and test vowels (claims C1, C6, C10 and C11 of docs/design/views/world.md).

Runs pyworld (WORLD's C++ code) on the synthetic vowels of
tools/check_world_claims.py, whose aperiodicity is known exactly, and on the
gallery sentence with its stored Harvest track. pyworld is used only here,
as a comparison oracle; it is not a sonore dependency.

    pip install pyworld            # development only
    python tools/crosscheck_world_vocoder.py
"""

from pathlib import Path

import numpy as np
import pyworld
import soundfile as sf
from check_world_claims import (
    FS,
    HOP,
    N_FFT,
    band_aperiodicity,
    cheaptrick,
    contour,
    d4c,
    phase_of,
    report,
    rising,
    true_band,
    vowel,
    world_synthesize,
)

SPEECH = Path(__file__).resolve().parents[1] / "docs" / "speech"
BANDS = [(0, 1000), (1000, 2000), (2000, 4000), (4000, 7000)]


def d4c_bands(x, f, times, bands=BANDS):
    """pyworld's D4C at the window times, as band aperiodicity: the power
    ratio ap^2 averaged over each band, weighted by CheapTrick's envelope."""
    x = np.ascontiguousarray(x)
    f0 = np.ascontiguousarray(f[np.round(times * FS).astype(int)])
    t = np.ascontiguousarray(times)
    sp = pyworld.cheaptrick(x, f0, t, int(FS))
    ap = pyworld.d4c(x, f0, t, int(FS))
    fr = np.fft.rfftfreq(N_FFT, 1 / FS)
    out = []
    for lo, hi in bands:
        m = (fr >= lo) & (fr < hi)
        out.append(10 * np.log10(np.sum(ap[:, m] ** 2 * sp[:, m]) / np.sum(sp[:, m])))
    return np.array(out), ap


def within_80_db(sp):
    """Where WORLD's envelope is within 80 dB of each time window's peak. Below
    that, the tiny random noise WORLD adds to keep logarithms finite, not
    the signal, sets its values."""
    return sp > 1e-8 * sp.max(axis=1, keepdims=True)


def main():
    print(f"pyworld {pyworld.__version__}")
    rng = np.random.default_rng(1)
    n = int(FS)
    fr = np.fft.rfftfreq(N_FFT, 1 / FS)

    # The port's envelope against WORLD's on the steady vowels
    for f0 in (100.0, 200.0, 300.0):
        f = contour("steady", n, f0)
        x = vowel(f, lambda q: np.full_like(np.asarray(q, float), -200.0), rng, noise=False)
        times = np.arange(0.2, 0.8, HOP)
        sp = pyworld.cheaptrick(x, f[np.round(times * FS).astype(int)], times, int(FS))
        ours = np.array([cheaptrick(x, t, f0) for t in times])
        d = 10 * np.log10(ours / sp)[within_80_db(sp)]
        report("C1", f"CheapTrick port vs pyworld, F0 {f0:.0f} Hz: largest |diff| [dB]", np.abs(d).max())

    # D4C on the vowels of known aperiodicity
    times = np.arange(0.1, 0.9, 2 * HOP)
    cases = {
        "flat -20 dB": lambda q: np.full_like(np.asarray(q, float), -20.0),
        "flat -6 dB": lambda q: np.full_like(np.asarray(q, float), -6.0),
        "rising -30 to -5 dB": rising,
    }
    for kind in ("steady", "vibrato"):
        f = contour(kind, n)
        for label, ap_db in cases.items():
            x = vowel(f, ap_db, rng)
            est, _ = d4c_bands(x, f, times)
            truth = true_band(ap_db, BANDS, f, times)
            print(f"     {kind}, {label}: truth {np.round(truth, 1)}, D4C {np.round(est, 1)}")
            err = est - truth
            report("C6", f"D4C, {kind}, {label}: worst band error [dB]", err[np.argmax(np.abs(err))])
        x = vowel(f, lambda q: np.full_like(np.asarray(q, float), -200.0), rng, noise=False)
        est, _ = d4c_bands(x, f, times)
        report("C6", f"D4C, {kind}, no noise: highest band estimate [dB]", est.max())

    # D4C at 16 kHz measures one band: its dB curve is two straight lines
    f = contour("vibrato", n)
    x = vowel(f, rising, rng)
    _, ap = d4c_bands(x, f, times)
    db = 20 * np.log10(ap)
    resid = max(
        np.abs(
            db[j]
            - np.interp(fr, [0, 3000, FS / 2], [db[j, 0], db[j, np.argmin(np.abs(fr - 3000))], db[j, -1]])
        ).max()
        for j in range(len(db))
    )
    report("C6", "D4C at 16 kHz: largest departure from straight lines through 0, 3k, 8 kHz [dB]", resid)
    report("C6", "D4C at 16 kHz: value at 0 Hz, every voiced time window [dB]", db[:, 0].max())
    report("C6", "D4C at 16 kHz: value at 8 kHz, every voiced time window [dB]", db[:, -1].min())

    # F0 errors: D4C against the harmonic residual
    truth = true_band(rising, BANDS, f, times)
    for rel in (0.003, 0.01, 0.03):
        g = f * (1 + rel)
        est, _ = d4c_bands(x, g, times)
        report(
            "C6",
            f"D4C, F0 {rel:.1%} high: worst band error [dB]",
            (est - truth)[np.argmax(np.abs(est - truth))],
        )
        est = band_aperiodicity(x, g, phase_of(g), times, BANDS)
        report(
            "C5",
            f"harmonic residual, F0 {rel:.1%} high: worst band error [dB]",
            (est - truth)[np.argmax(np.abs(est - truth))],
        )

    # The gallery sentence, with its stored Harvest track
    x, fs = sf.read(SPEECH / "bdl_arctic_a0131.flac")
    track = np.loadtxt(SPEECH / "bdl_arctic_a0131_f0.csv", delimiter=",", skiprows=2)
    t, f0 = np.ascontiguousarray(track[:, 0]), np.ascontiguousarray(track[:, 1])
    if fs != FS:
        raise SystemExit(f"expected {FS:g} Hz, got {fs}")
    sp = pyworld.cheaptrick(x, f0, t, fs)
    voiced = f0 > 0
    ours = np.array([cheaptrick(x, tt, ff) for tt, ff in zip(t[voiced], f0[voiced], strict=True)])
    d = 10 * np.log10(ours / sp[voiced])[within_80_db(sp[voiced])]
    report(
        "C1", "CheapTrick port vs pyworld on bdl, voiced time windows: largest |diff| [dB]", np.abs(d).max()
    )
    ap = pyworld.d4c(x, f0, t, fs)
    report(
        "C6",
        "D4C on bdl: share of voiced time windows left fully aperiodic (0 dB everywhere)",
        np.mean(np.all(ap[voiced] > 0.999, axis=1)),
    )

    # C10: the ports reproduce WORLD
    ours = d4c(x, t, f0)
    report(
        "C10",
        "D4C port vs pyworld on bdl, every time window: largest |diff| [dB]",
        np.abs(20 * np.log10(ours / ap)).max(),
    )
    y_world = pyworld.synthesize(f0, sp, ap, fs, HOP * 1000)
    y_ours = world_synthesize(f0, sp, ap, HOP * 1000, len(y_world))
    report(
        "C10",
        "synthesis port vs pyworld on bdl: largest |diff| re largest |sample|",
        np.abs(y_world - y_ours).max() / np.abs(y_world).max(),
    )
    for kind in ("steady", "vibrato"):
        f = contour(kind, n)
        x = vowel(f, rising, rng)
        times = np.arange(0.0, 1.0, HOP)
        f0v = np.ascontiguousarray(f[np.round(times * FS).astype(int)])
        ap = pyworld.d4c(x, f0v, times, int(FS))
        report(
            "C10",
            f"D4C port vs pyworld, {kind} vowel: largest |diff| [dB]",
            np.abs(20 * np.log10(d4c(x, times, f0v) / ap)).max(),
        )
        sp = pyworld.cheaptrick(x, f0v, times, int(FS))
        y_world = pyworld.synthesize(f0v, sp, ap, int(FS), HOP * 1000)
        y_ours = world_synthesize(f0v, sp, ap, HOP * 1000, len(y_world))
        report(
            "C10",
            f"synthesis port vs pyworld, {kind} vowel: largest |diff| re largest |sample|",
            np.abs(y_world - y_ours).max() / np.abs(y_world).max(),
        )

    # C11: WORLD's own round trip on the sentence: analyse, synthesize,
    # analyse the result with the same F0 track
    x, _ = sf.read(SPEECH / "bdl_arctic_a0131.flac")
    sp = pyworld.cheaptrick(x, f0, t, fs)
    ap = pyworld.d4c(x, f0, t, fs)
    y = np.ascontiguousarray(pyworld.synthesize(f0, sp, ap, fs, HOP * 1000)[: len(x)])
    sp2 = pyworld.cheaptrick(y, f0, t, fs)
    ap2 = pyworld.d4c(y, f0, t, fs)
    within_40_db = sp[voiced] > 1e-4 * sp[voiced].max(axis=1, keepdims=True)
    d = np.abs(10 * np.log10(sp2[voiced] / sp[voiced]))[within_40_db]
    report("C11", "WORLD round trip on bdl, envelope within 40 dB of peak: median |diff| [dB]", np.median(d))
    report(
        "C11",
        "WORLD round trip on bdl, envelope within 40 dB of peak: 95th pct |diff| [dB]",
        np.percentile(d, 95),
    )
    at_3k = np.argmin(np.abs(fr - 3000))
    d = np.abs(20 * np.log10(ap2[voiced, at_3k] / ap[voiced, at_3k]))
    report("C11", "WORLD round trip on bdl, D4C at 3 kHz: median |diff| [dB]", np.median(d))
    report("C11", "WORLD round trip on bdl, D4C at 3 kHz: 95th pct |diff| [dB]", np.percentile(d, 95))


if __name__ == "__main__":
    main()
