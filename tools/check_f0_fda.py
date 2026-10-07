"""The F0 design's prototype against laryngograph reference F0 (FDA database).

The FDA evaluation database (Bagshaw, CSTR, University of Edinburgh) has 50
sentences from a male (rl) and a female (sb) speaker, speech and
laryngograph at 20 kHz, and an F0 contour derived from the laryngograph's
pitch marks. It has no stated license, so it is not in this repository;
pass the folder that holds its rl/ and sb/ subfolders:

    python tools/check_f0_fda.py path/to/fda_eval

The speech is resampled to 16 kHz for the prototype and the cepstral
baseline. If pyworld is installed (development only), Harvest and DIO +
StoneMask are run as well, at 16 kHz and Harvest also at the native 20 kHz.
Prints voicing and F0 accuracy per speaker; see docs/design/views/f0.md.
"""

import sys
from pathlib import Path

import numpy as np
from check_f0_claims import F_HI, F_LO, FS, HOP, cepstral_track, scored, viterbi, window_times, yin_candidates
from scipy.signal import resample_poly

GUARD = 0.010  # time windows this close to a reference voicing boundary are not scored for voicing [s]


def read_fx(path):
    """Contour segments of an XMG .fx file: (time [s], F0 [Hz]) arrays, one per
    voiced stretch. The ASCII header ends at a form feed; '=' separates
    segments."""
    data = path.read_bytes()
    segs, cur = [], []
    for line in data[data.find(b"\x0c") + 1 :].decode().splitlines():
        line = line.strip()
        if line == "=":
            if cur:
                segs.append(np.array(cur))
            cur = []
        elif line:
            t_ms, f = line.split()[:2]
            cur.append((float(t_ms) / 1000, float(f)))
    if cur:
        segs.append(np.array(cur))
    return segs


def reference(segs, t):
    """Reference F0 at times t (0 outside every segment) and a mask of time windows
    far enough from a voicing boundary to score voicing."""
    f0 = np.zeros(len(t))
    edge = np.zeros(len(t), bool)
    for s in segs:
        inside = (t >= s[0, 0]) & (t <= s[-1, 0])
        f0[inside] = np.interp(t[inside], s[:, 0], s[:, 1])
        for b in (s[0, 0], s[-1, 0]):
            edge |= np.abs(t - b) < GUARD
    return f0, ~edge


def methods(x16, x20):
    t = window_times(len(x16))
    scored_windows = scored(x16, t, yin_candidates(x16, t))
    out = {f"tracker, threshold {th}": (t, viterbi(scored_windows, theta=th)) for th in (0.5, 0.4)}
    # Candidates offered at every difference-function minimum, so that only
    # the score decides voicing.
    scored_windows = scored(x16, t, yin_candidates(x16, t, d_max=1.0))
    for th in (0.5, 0.4):
        out[f"tracker, all minima, {th}"] = (t, viterbi(scored_windows, theta=th))
    out["cepstral baseline"] = cepstral_track(x16)
    try:
        import sonore as so
    except ImportError:
        pass
    else:
        trk = so.f0_track(so.Sound(x16, FS))
        out["so.f0_track"] = (trk.t, trk.f0[0])
    try:
        import pyworld
    except ImportError:
        return out
    kw = dict(f0_floor=F_LO, f0_ceil=F_HI, frame_period=HOP * 1000)
    x16c = np.ascontiguousarray(x16)
    f0, th = pyworld.harvest(x16c, int(FS), **kw)
    out["Harvest"] = (th, f0)
    f0, th = pyworld.harvest(np.ascontiguousarray(x20), 20000, **kw)
    out["Harvest, native 20 kHz"] = (th, f0)
    f0, th = pyworld.dio(x16c, int(FS), **kw)
    out["DIO + StoneMask"] = (th, pyworld.stonemask(x16c, f0, th, int(FS)))
    return out


def main(root):
    for spk in ("rl", "sb"):
        n = {}
        for sig in sorted((root / spk).glob("*.sig")):
            x20 = np.fromfile(sig, ">i2").astype(float) / 32768
            x16 = resample_poly(x20, 4, 5)
            segs = read_fx(sig.with_suffix(".fx"))
            for name, (t, est) in methods(x16, x20).items():
                ref, ok = reference(segs, t)
                rv, ev = ref > 0, est > 0
                rel = np.abs(est / np.where(rv, ref, 1) - 1)
                c = n.setdefault(name, {})
                for key, mask in [
                    ("voiced", rv & ok),  # time windows scored for voicing
                    ("missed", rv & ok & ~ev),
                    ("unvoiced", ~rv & ok),
                    ("false", ~rv & ok & ev),
                    ("ref", rv),
                    ("gross", rv & (~ev | (rel > 0.2))),  # unvoiced counts as an error
                    ("both", rv & ev),
                    ("both_gross", rv & ev & (rel > 0.2)),
                    ("both_5", rv & ev & (rel <= 0.05)),
                ]:
                    c[key] = c.get(key, 0) + int(np.sum(mask))
        print(f"speaker {spk}: {n[next(iter(n))]['ref']} reference-voiced time windows")
        for name, c in n.items():
            print(
                f"  {name:27s} misses {c['missed'] / c['voiced']:5.1%} of voiced, "
                f"voices {c['false'] / c['unvoiced']:5.1%} of unvoiced, "
                f"voicing error {(c['missed'] + c['false']) / (c['voiced'] + c['unvoiced']):5.1%} | "
                f"gross error {c['gross'] / c['ref']:5.1%}, "
                f"where both voice {c['both_gross'] / c['both']:5.2%} | "
                f"within 5% where both voice {c['both_5'] / c['both']:5.1%}"
            )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
