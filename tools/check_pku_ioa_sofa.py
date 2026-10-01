"""Hashes and a consistency check for the PKU-IOA SOFA files that ``so.load_hrirs`` downloads.

Run it where sofacoustics.org is reachable:

    python tools/check_pku_ioa_sofa.py                 # download, print SHA-256 lines for the registry
    python tools/check_pku_ioa_sofa.py --dat PKU-IOA/  # also compare with the original .dat files

The SHA-256 lines go into ``HRIR_DATABASES`` in src/sonore/stimuli/hrir_data.py.
With ``--dat``, every SOFA position must match a ``.dat`` position (as Cartesian
points, so a flipped azimuth convention shows up) and carry the same IRs.
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

import sonore as so
from sonore.stimuli import hrir_data


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dat", type=Path, help="folder holding the original PKU-IOA .dat files")
    parser.add_argument(
        "--cache", type=Path, help="where to keep the downloads (default: a temporary folder)"
    )
    args = parser.parse_args()

    db = hrir_data.HRIR_DATABASES["pku-ioa"]
    cache = args.cache or Path(tempfile.mkdtemp(prefix="pku-ioa-"))
    paths = {}
    print(f"Downloading {len(db.files)} SOFA files (about 13 MB each) into {cache}", file=sys.stderr)
    print("Checksums to paste into HRIR_DATABASES['pku-ioa'] (nothing to type here):")
    for d, (fname, pinned) in db.files.items():
        path = cache / fname
        if not path.exists():
            print(f"  downloading {fname} ...", file=sys.stderr, flush=True)
            hrir_data._download(db.base_url + fname, path)
        paths[d] = path
        digest = hrir_data._sha256(path)
        note = (
            "" if pinned is None else ("  # matches the pinned value" if digest == pinned else "  # DIFFERS")
        )
        print(f'            {d}: ("{fname}", "{digest}"),{note}')

    if args.dat is None:
        return
    dat = so.HRIRSet.from_pku_ioa(args.dat)
    key = {tuple(np.round(p, 4)): i for i, p in enumerate(dat.positions)}
    mirror = np.array([-1.0, 1.0, 1.0])  # left-right mirror image: x -> -x
    # Ways the SOFA copy could differ from the .dat files. For each, every SOFA
    # pair is fitted to the .dat pair with the best delay (within 64 samples) and
    # gain; the residual says whether that explanation holds.
    cases = {
        "same position, same ears": (False, False),
        "same position, ears swapped": (False, True),
        "mirrored azimuth, same ears": (True, False),
        "mirrored azimuth, ears swapped": (True, True),
    }
    fits = {c: [] for c in cases}
    raw, unmatched = [], 0
    for d, path in paths.items():
        sofa = so.HRIRSet.from_sofa(path)
        if sofa.fs != dat.fs:
            print(f"{d:>4} cm: sampling rate {sofa.fs:g} Hz, .dat files are {dat.fs:g} Hz")
        n = min(sofa.irs.shape[-1], dat.irs.shape[-1])
        for p, ir in zip(sofa.positions, sofa.irs, strict=True):
            i = key.get(tuple(np.round(p, 4)))
            j = key.get(tuple(np.round(p * mirror, 4)))
            if i is None or j is None:
                unmatched += 1
                continue
            raw.append(np.max(np.abs(ir[:, :n] - dat.irs[i, :, :n])) / np.max(np.abs(dat.irs[i])))
            for c, (flip, swap) in cases.items():
                ref = dat.irs[j if flip else i, :: -1 if swap else 1, :n]
                fits[c].append(fit(ir[:, :n], ref))
        print(f"{d:>4} cm: {len(sofa.positions)} positions, {sofa.irs.shape[-1]} taps")
    print(f"SOFA positions with no matching .dat position: {unmatched}")
    print(f"largest IR difference as stored, relative to the IR's peak: {max(raw):.3g}")
    print("best delay and gain per explanation (medians over positions; residual 0 = exact):")
    for c, f in fits.items():
        f = np.array(f)
        print(
            f"  {c:<32s} residual {np.median(f[:, 0]):.3g} (worst {f[:, 0].max():.3g}), "
            f"delay {np.median(f[:, 1]):+.0f} samples, gain {np.median(f[:, 2]):.4g}"
        )


def fit(a, b, max_lag=64):
    """Residual of ``a`` against ``b`` delayed and scaled to fit best, with the delay and gain.

    ``a`` and ``b`` are (2, n) pairs; one delay and gain serve both ears, so
    the interaural differences have to match as they are.
    """
    n = a.shape[-1]
    xc = np.fft.irfft(np.fft.rfft(a, 2 * n) * np.conj(np.fft.rfft(b, 2 * n)), 2 * n).sum(axis=0)
    lags = np.r_[0 : max_lag + 1, -max_lag:0]
    lag = int(lags[np.argmax(np.abs(xc[lags]))])
    bs = np.roll(np.pad(b, ((0, 0), (0, abs(lag)))), lag, axis=-1)[:, :n] if lag >= 0 else b[:, -lag:]
    bs = np.pad(bs, ((0, 0), (0, n - bs.shape[-1])))
    g = np.sum(a * bs) / max(np.sum(bs * bs), 1e-300)
    return np.linalg.norm(a - g * bs) / max(np.linalg.norm(a), 1e-300), lag, g


if __name__ == "__main__":
    main()
