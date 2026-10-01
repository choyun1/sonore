"""Hashes and a consistency check for the PKU-IOA SOFA files that ``so.load_hrirs`` downloads.

Run it where sofacoustics.org is reachable:

    python tools/check_pku_ioa_sofa.py                 # download, print SHA-256 lines for the registry
    python tools/check_pku_ioa_sofa.py --dat PKU-IOA/  # also compare with the original .dat files

The SHA-256 lines go into ``HRIR_DATABASES`` in src/sonore/stimuli/hrir_data.py.
With ``--dat``, every SOFA position must match a ``.dat`` position (as Cartesian
points, so a flipped azimuth convention shows up) and carry the same IRs.
"""

import argparse
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
    print("SHA-256 for HRIR_DATABASES['pku-ioa']:")
    for d, (fname, _) in db.files.items():
        path = cache / fname
        if not path.exists():
            hrir_data._download(db.base_url + fname, path)
        paths[d] = path
        print(f'            {d}: ("{fname}", "{hrir_data._sha256(path)}"),')

    if args.dat is None:
        return
    dat = so.HRIRSet.from_pku_ioa(args.dat)
    key = {tuple(np.round(p, 4)): i for i, p in enumerate(dat.positions)}
    worst_ir, unmatched = 0.0, 0
    for d, path in paths.items():
        sofa = so.HRIRSet.from_sofa(path)
        if sofa.fs != dat.fs:
            print(f"{d:>4} cm: sampling rate {sofa.fs:g} Hz, .dat files are {dat.fs:g} Hz")
        n = min(sofa.irs.shape[-1], dat.irs.shape[-1])
        for p, ir in zip(sofa.positions, sofa.irs, strict=True):
            i = key.get(tuple(np.round(p, 4)))
            if i is None:
                unmatched += 1
                continue
            err = np.max(np.abs(ir[:, :n] - dat.irs[i, :, :n])) / np.max(np.abs(dat.irs[i]))
            worst_ir = max(worst_ir, err)
        print(f"{d:>4} cm: {len(sofa.positions)} positions, {sofa.irs.shape[-1]} taps")
    print(f"SOFA positions with no matching .dat position: {unmatched}")
    print(f"largest IR difference, relative to the IR's peak: {worst_ir:.3g}")


if __name__ == "__main__":
    main()
