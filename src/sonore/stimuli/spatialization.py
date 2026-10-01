"""Head-related spatialization.

Coordinates
-----------
Cartesian positions are in meters, head-centered: **x = right, y = front,
z = up**. Head-centered "hcc" coordinates are ``(dist [cm], elev [deg],
azim [deg])`` with azimuth measured *clockwise* from the front (90° = right),
as in the PKU-IOA database. (SOFA files use counter-clockwise azimuth;
:meth:`HRIRSet.from_sofa` converts.)
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from fractions import Fraction
from functools import cached_property
from os import PathLike
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import fftconvolve, resample_poly
from scipy.spatial import ConvexHull, Delaunay

from sonore.core.sound import Sound

__all__ = [
    "rect_to_sph",
    "sph_to_rect",
    "rect_to_hcc",
    "hcc_to_rect",
    "linear_trajectory",
    "circular_trajectory",
    "distance_gain_db",
    "HRIRSet",
    "spatialize",
    "move_sound",
]


# ------------------------------------------------------------- coordinates
def rect_to_sph(x, y, z):
    """Cartesian -> (rho, theta=elevation, phi=angle from +x), radians."""
    hxy = np.hypot(x, y)
    return np.hypot(hxy, z), np.arctan2(z, hxy), np.arctan2(y, x)


def sph_to_rect(rho, theta, phi):
    return rho * np.cos(theta) * np.cos(phi), rho * np.cos(theta) * np.sin(phi), rho * np.sin(theta)


def rect_to_hcc(x, y, z):
    rho, theta, phi = rect_to_sph(x, y, z)
    return 100 * rho, np.degrees(theta), np.mod(90 - np.degrees(phi), 360)


def hcc_to_rect(dist, elev, azim):
    return sph_to_rect(np.asarray(dist) / 100, np.radians(elev), np.radians(90 - np.asarray(azim)))


def linear_trajectory(start: ArrayLike, end: ArrayLike, n: int) -> np.ndarray:
    """Straight line between two Cartesian points; shape ``(n, 3)``."""
    return np.linspace(np.asarray(start, float), np.asarray(end, float), n)


def circular_trajectory(start_hcc: ArrayLike, end_hcc: ArrayLike, n: int) -> np.ndarray:
    """Interpolate linearly in ``(dist, elev, azim)`` and return Cartesian
    points, shape ``(n, 3)``. E.g. ``(100, 0, 90) -> (100, 0, -90)`` sweeps
    from right to left through the front at 1 m."""
    hcc = np.linspace(np.asarray(start_hcc, float), np.asarray(end_hcc, float), n)
    return np.column_stack(hcc_to_rect(*hcc.T))


def distance_gain_db(distance: ArrayLike, ref: float = 1.0) -> np.ndarray:
    """Inverse-square-law level change [dB] for a point source re ``ref`` meters:
    ``-20*log10(d/ref)``, i.e. -6 dB per doubling of distance."""
    return -20 * np.log10(np.asarray(distance, float) / ref)


# ------------------------------------------------------------------ HRIRs
def _frac_shift(h: np.ndarray, shift: np.ndarray, n_out: int) -> np.ndarray:
    """Delay each IR in ``h`` (..., n) by ``shift`` (...,) samples (fractional ok)."""
    n = max(h.shape[-1], n_out) + int(np.ceil(np.max(np.abs(shift)))) + 64
    H = np.fft.rfft(h, n=n, axis=-1)
    f = np.fft.rfftfreq(n)
    H *= np.exp(-2j * np.pi * f * shift[..., None])
    return np.fft.irfft(H, n=n, axis=-1)[..., :n_out]


@dataclass
class HRIRSet:
    """A set of measured head-related impulse responses.

    Parameters
    ----------
    irs
        Shape ``(n_positions, 2, n_taps)`` (left, right).
    positions
        Cartesian source positions [m], shape ``(n_positions, 3)``.
    fs
        Sampling rate of the IRs.
    align
        Interpolate onset-aligned IRs and interpolate the onset delays
        separately. Avoids the comb filtering you get from averaging HRIRs with
        different delays.
    """

    irs: np.ndarray
    positions: np.ndarray
    fs: float
    align: bool = True
    onset_threshold_db: float = -20.0
    _pre: int = field(default=4, repr=False)

    def __post_init__(self):
        self.irs = np.asarray(self.irs, float)
        self.positions = np.asarray(self.positions, float)
        # drop duplicate positions (e.g. azimuth 0 and 360)
        _, keep = np.unique(np.round(self.positions, 6), axis=0, return_index=True)
        keep = np.sort(keep)
        self.irs, self.positions = self.irs[keep], self.positions[keep]

    def __repr__(self) -> str:
        return f"HRIRSet({len(self.positions)} positions, {self.irs.shape[-1]} taps, {self.fs:g} Hz)"

    # ---- loaders
    @classmethod
    def from_pku_ioa(cls, directory: str | PathLike, fs: float = 65536, **kwargs) -> HRIRSet:
        """Load the PKU-IOA database from its original ``.dat`` files
        (``azi{A}_elev{E}_dist{D}.dat``, float64, left then right), found in
        ``directory`` or any folder below it, such as the distributed
        ``dist{D}/elev{E}/`` layout. Empty files are skipped with a warning.
        :func:`load_hrirs` downloads the SOFA copy instead."""
        pattern = re.compile(r"azi(-?\d+)_elev(-?\d+)_dist(\d+)\.dat$")
        hcc, irs, paths, empty = [], [], [], []
        for p in sorted(Path(directory).rglob("*.dat")):
            m = pattern.search(p.name)
            if m:
                if p.stat().st_size == 0:
                    empty.append(p)
                    continue
                a, e, d = map(float, m.groups())
                hcc.append((d, e, a))
                irs.append(np.fromfile(p))
                paths.append(p)
        if empty:
            warnings.warn(
                f"skipped {len(empty)} empty .dat file(s), e.g. {empty[0]}; those directions are missing",
                stacklevel=2,
            )
        if not irs:
            raise FileNotFoundError(f"no PKU-IOA .dat files in {directory}")
        sizes = [x.size for x in irs]
        n = max(set(sizes), key=sizes.count)  # the usual length, 2048
        odd = [(p, k) for p, k in zip(paths, sizes, strict=True) if k != n]
        if odd:
            listed = "\n".join(f"  {p} ({k} values)" for p, k in odd[:10])
            raise ValueError(
                f"{len(odd)} of {len(irs)} .dat files are not {n} float64 values (left then right):"
                f"\n{listed}" + ("\n  ..." if len(odd) > 10 else "")
            )
        pos = np.column_stack(hcc_to_rect(*np.array(hcc).T))
        return cls(np.array(irs).reshape(len(irs), 2, -1), pos, fs, **kwargs)

    @classmethod
    def from_sofa(cls, path: str | PathLike, **kwargs) -> HRIRSet:
        """Load a SOFA ``SimpleFreeFieldHRIR`` file (needs ``h5py``)."""
        import h5py

        with h5py.File(path, "r") as f:
            irs = f["Data.IR"][()]  # (M, R, N)
            fs = float(np.ravel(f["Data.SamplingRate"][()])[0])
            pos = f["SourcePosition"][()]  # (M, 3)
            ptype = f["SourcePosition"].attrs.get("Type", b"spherical")
        ptype = ptype.decode() if isinstance(ptype, bytes) else str(ptype)
        if ptype.lower().startswith("cartesian"):
            # SOFA: x = front, y = left, z = up
            xyz = np.column_stack([-pos[:, 1], pos[:, 0], pos[:, 2]])
        else:
            az, el, r = pos.T  # SOFA azimuth is counter-clockwise from front
            xyz = np.column_stack(hcc_to_rect(100 * r, el, np.mod(-az, 360)))
        return cls(irs[:, :2, :], xyz, fs, **kwargs)

    @classmethod
    def concat(cls, sets, **kwargs) -> HRIRSet:
        """Merge sets measured at the same sampling rate and length, e.g. one
        per distance, into a single set. ``kwargs`` go to :class:`HRIRSet`."""
        sets = list(sets)
        if not sets:
            raise ValueError("no HRIR sets to merge")
        fs, taps = sets[0].fs, sets[0].irs.shape[1:]
        for s in sets[1:]:
            if s.fs != fs or s.irs.shape[1:] != taps:
                raise ValueError("HRIR sets differ in sampling rate or IR shape; resample before merging")
        irs = np.concatenate([s.irs for s in sets])
        positions = np.concatenate([s.positions for s in sets])
        return cls(irs, positions, fs, **kwargs)

    # ---- interpolation
    @cached_property
    def _onsets(self) -> np.ndarray:
        env = np.abs(self.irs)
        thresh = env.max(axis=-1, keepdims=True) * 10 ** (self.onset_threshold_db / 20)
        return np.argmax(env >= thresh, axis=-1).astype(float)  # (M, 2)

    @cached_property
    def _aligned(self) -> np.ndarray:
        shift = self._pre - self._onsets
        return _frac_shift(self.irs, shift, self.irs.shape[-1])

    @cached_property
    def _single_distance(self) -> bool:
        r = np.linalg.norm(self.positions, axis=1)
        return np.ptp(r) < 1e-3 * r.mean()

    @cached_property
    def _triangulation(self):
        if self._single_distance:
            u = self.positions / np.linalg.norm(self.positions, axis=1, keepdims=True)
            hull = ConvexHull(u)
            simplices = hull.simplices
            inv = np.linalg.inv(u[simplices].transpose(0, 2, 1))  # (T, 3, 3)
            return simplices, inv
        return Delaunay(self.positions)

    def _weights(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Barycentric weights: (indices (N, k), weights (N, k))."""
        if self._single_distance:
            simplices, inv = self._triangulation
            u = points / np.linalg.norm(points, axis=1, keepdims=True)
            w = np.einsum("tij,nj->nti", inv, u)  # (N, T, 3)
            ok = np.all(w >= -1e-9, axis=-1)
            if not ok.any(axis=1).all():
                raise ValueError("some directions are outside the measured sphere")
            tri = np.argmax(ok, axis=1)
            w = w[np.arange(len(points)), tri]
            return simplices[tri], w / w.sum(axis=1, keepdims=True)
        tess = self._triangulation
        simp = tess.find_simplex(points)
        if np.any(simp < 0):
            raise ValueError("some positions are outside the measured region; cannot interpolate")
        T = tess.transform[simp]
        b = np.einsum("nij,nj->ni", T[:, :3, :], points - T[:, 3, :])
        w = np.column_stack([b, 1 - b.sum(axis=1)])
        return tess.simplices[simp], w

    def at(self, positions: ArrayLike, fs: float | None = None) -> np.ndarray:
        """Interpolated HRIRs at Cartesian ``positions`` (N, 3), resampled to
        ``fs``. Returns shape ``(N, 2, n_taps)``."""
        points = np.atleast_2d(np.asarray(positions, float))
        idx, w = self._weights(points)
        if self.align:
            shapes = np.einsum("nk,nkcs->ncs", w, self._aligned[idx])
            onsets = np.einsum("nk,nkc->nc", w, self._onsets[idx])
            n_out = self.irs.shape[-1] + int(np.ceil(self._onsets.max()))
            h = _frac_shift(shapes, onsets - self._pre, n_out)
        else:
            h = np.einsum("nk,nkcs->ncs", w, self.irs[idx])
        if fs is not None and fs != self.fs:
            r = Fraction(fs / self.fs).limit_denominator(10000)
            h = resample_poly(h, r.numerator, r.denominator, axis=-1)
        return h


def spatialize(sound: Sound, position: ArrayLike, hrirs: HRIRSet) -> Sound:
    """Render a mono sound at a fixed Cartesian position."""
    h = hrirs.at(position, fs=sound.fs)[0]
    return sound.mono().convolve(Sound(h.T, sound.fs))


def move_sound(sound: Sound, trajectory: np.ndarray, hrirs: HRIRSet) -> Sound:
    """Render a mono sound moving along ``trajectory`` (N, 3), traversed at a
    constant rate over the sound's duration.

    Each of the N positions gets a raised-cosine weighting window centered on
    its moment in time; neighbouring windows cross-fade and sum to exactly 1,
    and each windowed segment is convolved with its (interpolated) HRIR.
    """
    trajectory = np.atleast_2d(trajectory)
    if len(trajectory) == 1:
        return spatialize(sound, trajectory[0], hrirs)
    x = sound.mono().data[:, 0]
    n, N = len(x), len(trajectory)
    H = hrirs.at(trajectory, fs=sound.fs)  # (N, 2, taps)
    out = np.zeros((n + H.shape[-1] - 1, 2))
    knots = np.linspace(0, n - 1, N)
    for i in range(N):
        lo = int(np.floor(knots[i - 1])) if i > 0 else 0
        hi = int(np.ceil(knots[i + 1])) + 1 if i < N - 1 else n
        s = np.arange(lo, hi)
        u = np.interp(s, knots, np.arange(N)) - i  # position in "knot units", in [-1, 1]
        w = np.where(np.abs(u) < 1, (1 + np.cos(np.pi * u)) / 2, 0.0)
        if i == 0:
            w[s <= knots[0]] = 1
        if i == N - 1:
            w[s >= knots[-1]] = 1
        seg = fftconvolve((x[lo:hi] * w)[:, None], H[i].T, axes=0)
        out[lo : lo + len(seg)] += seg
    return Sound(out, sound.fs)
