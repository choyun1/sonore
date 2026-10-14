"""Numerical checks for docs/design/spatial/elevation.md: what PKU-IOA's HRIRs hold
across elevation, how well sonore interpolates between its elevation rings, and what
`HRIRSet.at` does below the lowest ring. Each line prints what was measured.

    python tools/check_elevation_claims.py --pku DIR

DIR holds the PKU-IOA HRIRs: the original .dat files (any layout). Only the 1 m
files are read. Without --pku it uses so.load_hrirs(distances=(100,)), which
downloads them. It takes a few seconds.

Like tools/check_moving_trajectories.py, this runs sonore itself: it measures what
sonore's interpolation does, not an independent reimplementation of it.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np

import sonore as so

DISTANCE = 100  # cm
N_FFT = 4096
BAND = (1000.0, 16000.0)  # Hz, where the spectral differences are compared


def report(text, value, unit=""):
    print(f"{text:<78s} {value:>9.4g} {unit}")


def load(pku):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if pku is None:
            return so.load_hrirs(distances=(DISTANCE,))
        directory = Path(pku)
        if not any(directory.rglob(f"*_dist{DISTANCE}.dat")):
            raise SystemExit(f"no *_dist{DISTANCE}.dat files under {directory}")
        return so.HRIRSet.from_pku_ioa(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pku", help="directory with the PKU-IOA .dat files")
    args = parser.parse_args()
    loaded = load(args.pku)
    at_distance = np.isclose(np.linalg.norm(loaded.positions, axis=1), DISTANCE / 100, rtol=1e-3)
    hrirs = so.HRIRSet(loaded.irs[at_distance], loaded.positions[at_distance], loaded.fs)
    fs = hrirs.fs
    freqs = np.fft.rfftfreq(N_FFT, 1 / fs)
    band = (freqs >= BAND[0]) & (freqs <= BAND[1])

    def spectrum_db(irs):
        return 20 * np.log10(np.abs(np.fft.rfft(irs, N_FFT, axis=-1)) + 1e-12)

    def spectral_distance(a, b):
        """RMS difference [dB] of the magnitude spectra over BAND, both ears, per position."""
        difference = spectrum_db(a)[..., band] - spectrum_db(b)[..., band]
        return np.sqrt(np.mean(difference**2, axis=(-1, -2)))

    hcc = np.column_stack(so.rect_to_hcc(*hrirs.positions.T))
    elevation = np.round(hcc[:, 1], 3)
    azimuth = np.round(np.mod(hcc[:, 2], 360)) % 360
    n_taps = hrirs.irs.shape[-1]

    def without(keep):
        return so.HRIRSet(hrirs.irs[keep], hrirs.positions[keep], fs)

    def index(elev, azim):
        return np.flatnonzero((elevation == elev) & (azimuth == azim % 360))[0]

    print(f"PKU-IOA at {DISTANCE} cm: {len(hrirs.positions)} directions, {fs:g} Hz")

    print("\nCoverage")
    rings = np.unique(elevation)
    for ring in rings:
        on_ring = np.sort(azimuth[elevation == ring])
        step = np.min(np.diff(on_ring)) if len(on_ring) > 1 else float("nan")
        report(f"  ring at {ring:+.0f} deg: directions (azimuth step, deg)", len(on_ring), f"({step:g})")
    report("  lowest measured elevation", rings.min(), "deg")

    print("\nCues in the median plane, against one side (both at 0 deg elevation)")

    def interaural(irs):
        left, right = irs
        lag = np.argmax(np.correlate(left, right, "full")) - (len(left) - 1)
        return abs(lag) / fs * 1e6, abs(10 * np.log10(np.sum(right**2) / np.sum(left**2)))

    median_plane = [index(e, a) for a in (0, 180) for e in rings if np.any((elevation == e) & (azimuth == a))]
    itds, ilds = zip(*(interaural(hrirs.irs[i]) for i in median_plane), strict=True)
    report("  median plane (front and back, every ring): largest |ITD|", max(itds), "us")
    report("  median plane: largest |ILD|, broadband", max(ilds), "dB")
    side_itd, side_ild = interaural(hrirs.irs[index(0, 90)])
    report("  90 deg right: |ITD|", side_itd, "us")
    report("  90 deg right: |ILD|, broadband", side_ild, "dB")

    print(f"\nSpectral distance, RMS dB over {BAND[0]:g}-{BAND[1]:g} Hz, both ears, median over directions")
    pairs = [
        (i, index(elevation[i] - 10, azimuth[i]))
        for i in range(len(elevation))
        if elevation[i] - 10 in rings and elevation[i] <= 50
    ]
    report(
        "  measured neighbors 10 deg apart in elevation (rings -40 to 50)",
        np.median([spectral_distance(hrirs.irs[a], hrirs.irs[b]) for a, b in pairs]),
        "dB",
    )
    pairs = [(i, index(elevation[i], azimuth[i] + 5)) for i in range(len(elevation)) if elevation[i] <= 50]
    report(
        "  measured neighbors 5 deg apart in azimuth (rings -40 to 50)",
        np.median([spectral_distance(hrirs.irs[a], hrirs.irs[b]) for a, b in pairs]),
        "dB",
    )

    print("\nInterpolation, one ring left out (gap 20 deg in elevation)")
    for ring in rings[(rings > rings.min()) & (rings <= 50)]:
        left_out = np.flatnonzero(elevation == ring)
        interpolated = without(elevation != ring).at(hrirs.positions[left_out])[..., :n_taps]
        below = [index(ring - 10, azimuth[i]) for i in left_out]
        report(
            f"  ring {ring:+.0f}: from rings {ring - 10:+.0f} and {ring + 10:+.0f}, against measured",
            np.median(spectral_distance(interpolated, hrirs.irs[left_out])),
            "dB",
        )
        report(
            "      (the measured ring 10 deg below, for comparison)",
            np.median(spectral_distance(hrirs.irs[below], hrirs.irs[left_out])),
            "dB",
        )
    odd = (azimuth % 10 == 5) & (elevation <= 50)
    left_out = np.flatnonzero(odd)
    interpolated = without(~odd).at(hrirs.positions[left_out])[..., :n_taps]
    report(
        "  every other azimuth left out (gap 10 deg in azimuth), rings -40 to 50",
        np.median(spectral_distance(interpolated, hrirs.irs[left_out])),
        "dB",
    )

    print("\nBelow the lowest ring")
    _, triangles, _ = hrirs._shells[0]
    unit = hrirs.positions / np.linalg.norm(hrirs.positions, axis=1, keepdims=True)

    def angle(i, j):
        return np.degrees(np.arccos(np.clip(np.sum(unit[i] * unit[j], axis=-1), -1, 1)))

    longest = np.max([angle(triangles[:, a], triangles[:, b]) for a, b in ((0, 1), (1, 2), (0, 2))], axis=0)
    cap = np.all(elevation[triangles] == rings.min(), axis=1)
    report("  triangles of the convex hull", len(triangles))
    report("  of them with every corner on the lowest ring (the hull's bottom face)", cap.sum())
    report("  longest side of any other triangle", longest[~cap].max(), "deg")
    report("  median longest side, all triangles", np.median(longest), "deg")
    report("  longest side of a bottom-face triangle", longest[cap].max(), "deg")
    corner_counts = np.bincount(triangles[cap].ravel(), minlength=len(unit))
    hub = np.argmax(corner_counts)
    report(f"  bottom-face triangles sharing one corner (azimuth {azimuth[hub]:g} deg)", corner_counts[hub])
    for elev, azim in ((-45, 0), (-50, 90), (-60, 0), (-90, 0)):
        point = np.column_stack(so.hcc_to_rect(DISTANCE, elev, azim))
        corners, weights = hrirs._weights(point)
        spread = np.abs((azimuth[corners[0]] - azim + 180) % 360 - 180)
        blended = ", ".join(f"{a:g}" for a in azimuth[corners[0]])
        report(
            f"  at ({elev:+d}, {azim:d}): weight on corners >= 90 deg away in azimuth ({blended})",
            weights[0][spread >= 90].sum(),
        )

    print("\nA path through the median plane")
    sweep = so.hcc_trajectory(dist=DISTANCE, elev=([0, 3], [rings.min(), 90]), azim=0)
    rendered = so.move_sound(so.gaussian_noise(3.0, 16000, rng=1), sweep, hrirs)
    report("  3 s of noise from the lowest ring to overhead, rendered: seconds", rendered.duration, "s")


if __name__ == "__main__":
    main()
