"""Head-related spatialization.

Coordinates
-----------
Cartesian positions are in meters, head-centered: **x = right, y = front,
z = up**. Head-centered "hcc" coordinates are ``(dist [cm], elev [deg],
azim [deg])`` with azimuth measured *clockwise* from the front (90° = right),
as in the PKU-IOA database. (SOFA files use counter-clockwise azimuth;
:meth:`HRIRSet.from_sofa` converts.)

Moving sources
--------------
A path is where the source is as a function of the time it emits, in
seconds. A sample emitted at time t_e reaches each ear after that ear's
delay d(t_e), the onset of the HRIR at the source's position. Measured
HRIRs such as PKU-IOA's keep the travel time from the source, so this
delay is the propagation delay plus the interaural delay. :func:`move_sound`
solves t = t_e + d(t_e) for every output sample t and reads the sound at
t_e between samples, so the delay changes smoothly, and the Doppler shift
of a source that changes distance comes out of the same read. The rest of
the HRIR, its shape with the onset removed, is switched every ``hop``.
Beyond the measured distances, distance acts only through travel time and
level: the HRIR of the nearest measured distance in the same direction,
delayed by the extra distance over the speed of sound and scaled by the
ratio of distances (1/r).
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from functools import cached_property
from os import PathLike
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import CubicSpline
from scipy.signal import fftconvolve, minimum_phase, resample_poly
from scipy.spatial import ConvexHull
from scipy.special import i0

from sonore.core.sound import Sound
from sonore.signals.processing import _track

__all__ = [
    "rect_to_sph",
    "sph_to_rect",
    "rect_to_hcc",
    "hcc_to_rect",
    "linear_trajectory",
    "circular_trajectory",
    "hcc_trajectory",
    "distance_gain_db",
    "HRIRSet",
    "spatialize",
    "move_sound",
    "SPEED_OF_SOUND",
]

SPEED_OF_SOUND = 343.0  # m/s, air at about 20 degrees C

# A path: Cartesian positions [m], shape (n, 3), at emission times t [s], shape (n,).
SourcePath = Callable[[np.ndarray], np.ndarray]


# ------------------------------------------------------------- coordinates
def rect_to_sph(x, y, z):
    """Cartesian -> (rho, theta=elevation, phi=angle from +x), radians."""
    horizontal_dist = np.hypot(x, y)
    return np.hypot(horizontal_dist, z), np.arctan2(z, horizontal_dist), np.arctan2(y, x)


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


def hcc_trajectory(dist, elev, azim) -> SourcePath:
    """A path in head-centered coordinates, as a function of time.

    Each coordinate is a number, a ``(times, values)`` pair (interpolated
    linearly, held before the first and after the last time), or a function
    of time in seconds. ``dist`` is in cm, ``elev`` and ``azim`` in degrees
    (azimuth clockwise from the front). Interpolating in these coordinates,
    rather than between Cartesian points, keeps an arc an arc.

    Returns a function ``t -> positions``, Cartesian [m], shape ``(n, 3)``,
    which :func:`move_sound` takes as a trajectory. For example, the azimuth
    swing of Cho & Kidd (2022), 30 degrees to either side at 2 Hz, 1 m away::

        so.hcc_trajectory(100, 0, lambda t: 30 * np.sin(2 * np.pi * 2 * t))
    """
    coordinates = {"dist": dist, "elev": elev, "azim": azim}

    def value_at(name: str, t: np.ndarray) -> np.ndarray:
        value = coordinates[name]
        if callable(value):
            return np.broadcast_to(np.asarray(value(t), float), t.shape)
        return np.broadcast_to(_track(value, t, name), t.shape)

    def path(t: ArrayLike) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, float))
        return np.column_stack(hcc_to_rect(value_at("dist", t), value_at("elev", t), value_at("azim", t)))

    return path


def _as_path(trajectory, duration: float) -> SourcePath:
    """Any trajectory form as a path: a function of time, a ``(times, points)``
    pair, or an ``(N, 3)`` array of points spread evenly over ``duration``.
    Points are interpolated linearly in Cartesian coordinates."""
    if callable(trajectory):
        return lambda t: np.atleast_2d(np.asarray(trajectory(np.atleast_1d(np.asarray(t, float))), float))
    if isinstance(trajectory, tuple) and len(trajectory) == 2 and np.ndim(trajectory[1]) == 2:
        times, points = (np.asarray(part, float) for part in trajectory)
        if times.ndim != 1 or points.shape != (len(times), 3):
            raise ValueError(
                "a (times, points) trajectory needs (N,) times and (N, 3) points, "
                f"got {times.shape} and {points.shape}"
            )
        if np.any(np.diff(times) <= 0):
            raise ValueError("the trajectory's times must increase")
    else:
        points = np.atleast_2d(np.asarray(trajectory, float))
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"trajectory points must have shape (N, 3), got {points.shape}")
        times = np.linspace(0, duration, len(points))

    def path(t: ArrayLike) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, float))
        return np.column_stack([np.interp(t, times, points[:, axis]) for axis in range(3)])

    return path


def distance_gain_db(distance: ArrayLike, ref: float = 1.0) -> np.ndarray:
    """Inverse-square-law level change [dB] for a point source re ``ref`` meters:
    ``-20*log10(d/ref)``, i.e. -6 dB per doubling of distance."""
    return -20 * np.log10(np.asarray(distance, float) / ref)


# ------------------------------------------------------------------ HRIRs
def _frac_shift(h: np.ndarray, shift: np.ndarray, n_out: int) -> np.ndarray:
    """Delay each IR in ``h`` (..., n) by ``shift`` (...,) samples (fractional ok)."""
    n_fft = max(h.shape[-1], n_out) + int(np.ceil(np.max(np.abs(shift)))) + 64
    spectrum = np.fft.rfft(h, n=n_fft, axis=-1)
    freqs = np.fft.rfftfreq(n_fft)
    spectrum *= np.exp(-2j * np.pi * freqs * shift[..., None])
    return np.fft.irfft(spectrum, n=n_fft, axis=-1)[..., :n_out]


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
        for path in sorted(Path(directory).rglob("*.dat")):
            match = pattern.search(path.name)
            if match:
                if path.stat().st_size == 0:
                    empty.append(path)
                    continue
                azim, elev, dist = map(float, match.groups())
                hcc.append((dist, elev, azim))
                irs.append(np.fromfile(path))
                paths.append(path)
        if empty:
            warnings.warn(
                f"skipped {len(empty)} empty .dat file(s), e.g. {empty[0]}; those directions are missing",
                stacklevel=2,
            )
        if not irs:
            raise FileNotFoundError(f"no PKU-IOA .dat files in {directory}")
        sizes = [x.size for x in irs]
        usual_size = max(set(sizes), key=sizes.count)  # the usual length, 2048
        wrong_sized = [(path, size) for path, size in zip(paths, sizes, strict=True) if size != usual_size]
        if wrong_sized:
            listed = "\n".join(f"  {path} ({size} values)" for path, size in wrong_sized[:10])
            raise ValueError(
                f"{len(wrong_sized)} of {len(irs)} .dat files are not {usual_size} float64 values "
                "(left then right):"
                f"\n{listed}" + ("\n  ..." if len(wrong_sized) > 10 else "")
            )
        positions = np.column_stack(hcc_to_rect(*np.array(hcc).T))
        return cls(np.array(irs).reshape(len(irs), 2, -1), positions, fs, **kwargs)

    @classmethod
    def from_sofa(cls, path: str | PathLike, **kwargs) -> HRIRSet:
        """Load a SOFA ``SimpleFreeFieldHRIR`` file (needs ``h5py``)."""
        import h5py

        with h5py.File(path, "r") as sofa:
            irs = sofa["Data.IR"][()]  # (M, R, N)
            fs = float(np.ravel(sofa["Data.SamplingRate"][()])[0])
            source_pos = sofa["SourcePosition"][()]  # (M, 3)
            pos_type = sofa["SourcePosition"].attrs.get("Type", b"spherical")
        pos_type = pos_type.decode() if isinstance(pos_type, bytes) else str(pos_type)
        if pos_type.lower().startswith("cartesian"):
            # SOFA: x = front, y = left, z = up
            xyz = np.column_stack([-source_pos[:, 1], source_pos[:, 0], source_pos[:, 2]])
        else:
            azimuth, elevation, radius = source_pos.T  # SOFA azimuth is counter-clockwise from front
            xyz = np.column_stack(hcc_to_rect(100 * radius, elevation, np.mod(-azimuth, 360)))
        return cls(irs[:, :2, :], xyz, fs, **kwargs)

    @classmethod
    def concat(cls, sets, **kwargs) -> HRIRSet:
        """Merge sets measured at the same sampling rate and length, e.g. one
        per distance, into a single set. ``kwargs`` go to :class:`HRIRSet`."""
        sets = list(sets)
        if not sets:
            raise ValueError("no HRIR sets to merge")
        fs, ir_shape = sets[0].fs, sets[0].irs.shape[1:]
        for hrir_set in sets[1:]:
            if hrir_set.fs != fs or hrir_set.irs.shape[1:] != ir_shape:
                raise ValueError("HRIR sets differ in sampling rate or IR shape; resample before merging")
        irs = np.concatenate([hrir_set.irs for hrir_set in sets])
        positions = np.concatenate([hrir_set.positions for hrir_set in sets])
        return cls(irs, positions, fs, **kwargs)

    # ---- interpolation
    @cached_property
    def _onsets(self) -> np.ndarray:
        envelope = np.abs(self.irs)
        threshold = envelope.max(axis=-1, keepdims=True) * 10 ** (self.onset_threshold_db / 20)
        return np.argmax(envelope >= threshold, axis=-1).astype(float)  # (M, 2)

    @cached_property
    def _aligned(self) -> np.ndarray:
        shift = self._pre - self._onsets
        return _frac_shift(self.irs, shift, self.irs.shape[-1])

    @cached_property
    def _shells(self) -> list[tuple[float, np.ndarray, np.ndarray]]:
        """The measured distances, nearest first: (radius [m], triangles of
        position indices (T, 3), their inverse direction matrices (T, 3, 3))."""
        radii = np.linalg.norm(self.positions, axis=1)
        order = np.argsort(radii)
        # a new shell starts wherever the radius jumps by more than 0.1%
        breaks = np.flatnonzero(np.diff(radii[order]) > 1e-3 * radii[order][1:]) + 1
        shells = []
        for members in np.split(order, breaks):
            directions = self.positions[members] / radii[members, None]
            triangles = members[ConvexHull(directions).simplices]
            unit = self.positions[triangles] / radii[triangles][..., None]  # (T, 3 corners, 3)
            shells.append((float(radii[members].mean()), triangles, np.linalg.inv(unit.transpose(0, 2, 1))))
        return shells

    @property
    def _single_distance(self) -> bool:
        return len(self._shells) == 1

    @property
    def distances(self) -> np.ndarray:
        """The measured distances [m], nearest first."""
        return np.array([radius for radius, _, _ in self._shells])

    def _direction_weights(self, shell: int, directions: np.ndarray, chunk: int = 256):
        """Spherical barycentric weights on one shell: (indices (N, 3), weights (N, 3))."""
        _, triangles, inverse = self._shells[shell]
        indices = np.empty((len(directions), 3), int)
        weights = np.empty((len(directions), 3))
        for start in range(0, len(directions), chunk):
            block = slice(start, start + chunk)
            candidate = np.einsum("tij,nj->nti", inverse, directions[block])  # (n, T, 3)
            inside = np.all(candidate >= -1e-9, axis=-1)
            if not inside.any(axis=1).all():
                raise ValueError("some directions are outside the measured sphere")
            triangle = np.argmax(inside, axis=1)
            found = candidate[np.arange(len(triangle)), triangle]
            indices[block] = triangles[triangle]
            weights[block] = found / found.sum(axis=1, keepdims=True)
        return indices, weights

    def _weights(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Interpolation weights: (indices (N, k), weights (N, k)). On one
        shell, by direction only; between shells, by direction on the two
        shells around the point's distance, then linearly in distance."""
        directions = points / np.linalg.norm(points, axis=1, keepdims=True)
        if self._single_distance:
            return self._direction_weights(0, directions)
        distances = self.distances
        radii = np.linalg.norm(points, axis=1)
        if np.any(radii < distances[0] * (1 - 1e-6)) or np.any(radii > distances[-1] * (1 + 1e-6)):
            raise ValueError(
                "some positions are outside the measured distances "
                f"({distances[0]:g} to {distances[-1]:g} m); cannot interpolate"
            )
        upper = np.clip(np.searchsorted(distances, radii), 1, len(distances) - 1)
        fraction = np.clip((radii - distances[upper - 1]) / (distances[upper] - distances[upper - 1]), 0, 1)
        indices = np.empty((len(points), 6), int)
        weights = np.empty((len(points), 6))
        for shell in np.unique(np.concatenate([upper - 1, upper])):
            for column, side, share in ((0, upper - 1, 1 - fraction), (3, upper, fraction)):
                rows = np.flatnonzero(side == shell)
                if len(rows):
                    shell_indices, shell_weights = self._direction_weights(shell, directions[rows])
                    indices[rows, column : column + 3] = shell_indices
                    weights[rows, column : column + 3] = shell_weights * share[rows, None]
        return indices, weights

    def _nearest_measured(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Each point moved along its direction to the nearest measured distance,
        and its own distance [m]. Points between measured distances stay put; with
        several distances, points nearer than the nearest are left to fail."""
        distances = self.distances
        radii = np.linalg.norm(points, axis=1)
        lower = distances[0] if self._single_distance else 0.0
        measured = np.clip(radii, lower, distances[-1])
        return points * (measured / radii)[:, None], radii

    def _shapes(self, points: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
        """Onset-aligned HRIR shapes at ``points`` (N, 3), resampled to ``fs``,
        each with its onset ``_pre`` samples (at ``self.fs``) after the start,
        and their onsets [s], shape (N, 2)."""
        indices, weights = self._weights(points)
        shapes = np.einsum("nk,nkcs->ncs", weights, self._aligned[indices])
        onsets = np.einsum("nk,nkc->nc", weights, self._onsets[indices]) / self.fs
        if fs != self.fs:
            ratio = Fraction(fs / self.fs).limit_denominator(10000)
            shapes = resample_poly(shapes, ratio.numerator, ratio.denominator, axis=-1)
        return shapes, onsets

    def _onset_times(self, points: np.ndarray) -> np.ndarray:
        """Interpolated onsets [s] at ``points`` (N, 3), shape (N, 2)."""
        indices, weights = self._weights(points)
        return np.einsum("nk,nkc->nc", weights, self._onsets[indices]) / self.fs

    def _diffuse_field(self, fs: float) -> tuple[float, np.ndarray]:
        """What sound arriving from every direction at once sounds like at the ears,
        at the measured distance nearest 1 m, at ``fs``: the energy of one ear's HRIR
        averaged over directions and ears (scaled by 1/r to 1 m if that distance is
        not 1 m), and a minimum-phase filter per ear, shape (2, taps), whose power
        response is the HRIRs' power response averaged over directions."""
        radii = np.linalg.norm(self.positions, axis=1)
        nearest = self.distances[np.argmin(np.abs(self.distances - 1.0))]
        hrir = self.at(self.positions[np.abs(radii - nearest) < 1e-3 * nearest], fs=fs) * nearest
        energy = float(np.mean(np.sum(hrir**2, axis=-1)))
        n_fft = 2 * int(2 ** np.ceil(np.log2(hrir.shape[-1])))
        power = np.mean(np.abs(np.fft.rfft(hrir, n_fft, axis=-1)) ** 2, axis=0)  # (2, bins)
        # A linear-phase filter with this power response; its minimum-phase version
        # has the square root of it as magnitude, the average magnitude we want.
        # The first sample, the lag of half the FFT length, is dropped (the window is zero there)
        # so the filter is symmetric about its middle sample, as minimum_phase expects.
        linear_phase = (
            np.fft.fftshift(np.fft.irfft(power, n_fft, axis=-1), axes=-1)[:, 1:] * np.hanning(n_fft + 1)[1:-1]
        )
        filters = np.array([minimum_phase(linear_phase[ear], method="homomorphic") for ear in range(2)])
        return energy, filters

    def at(self, positions: ArrayLike, fs: float | None = None) -> np.ndarray:
        """Interpolated HRIRs at Cartesian ``positions`` (N, 3), resampled to
        ``fs``. Returns shape ``(N, 2, n_taps)``."""
        points = np.atleast_2d(np.asarray(positions, float))
        indices, weights = self._weights(points)
        if self.align:
            shapes = np.einsum("nk,nkcs->ncs", weights, self._aligned[indices])
            onsets = np.einsum("nk,nkc->nc", weights, self._onsets[indices])
            n_out = self.irs.shape[-1] + int(np.ceil(self._onsets.max()))
            hrir = _frac_shift(shapes, onsets - self._pre, n_out)
        else:
            hrir = np.einsum("nk,nkcs->ncs", weights, self.irs[indices])
        if fs is not None and fs != self.fs:
            ratio = Fraction(fs / self.fs).limit_denominator(10000)
            hrir = resample_poly(hrir, ratio.numerator, ratio.denominator, axis=-1)
        return hrir


def spatialize(
    sound: Sound, position: ArrayLike, hrirs: HRIRSet, speed_of_sound: float = SPEED_OF_SOUND
) -> Sound:
    """Render a mono sound at a fixed Cartesian position [m].

    Beyond the measured distances (or off the one distance of a
    single-distance set) the HRIR of the nearest measured distance in the
    same direction is delayed by the extra distance over ``speed_of_sound``
    and scaled by the ratio of distances (1/r).
    """
    point = np.atleast_2d(np.asarray(position, float))
    measured_point, radius = hrirs._nearest_measured(point)
    measured_radius = np.linalg.norm(measured_point)
    hrir = hrirs.at(measured_point, fs=sound.fs)[0]
    sound = sound.mono()
    if not np.isclose(radius[0], measured_radius):
        extra_delay = (radius[0] - measured_radius) / speed_of_sound * sound.fs
        n_out = len(sound) + max(0, int(np.ceil(extra_delay)))
        delayed = _read_between_samples(sound.data[:, 0], np.arange(n_out) - extra_delay)
        sound = Sound(delayed * (measured_radius / radius[0]), sound.fs)
    return sound.convolve(Sound(hrir.T, sound.fs))


_READ_HALF_WIDTH = 16  # the windowed sinc spans 32 samples
_READ_KAISER_BETA = 8.0
_DELAY_STEP = 1e-3  # s; spacing at which ear delays are looked up before smoothing


def _read_between_samples(signal: np.ndarray, positions: np.ndarray, chunk: int = 32768) -> np.ndarray:
    """``signal`` read at fractional sample ``positions`` with a 32-sample
    Kaiser-windowed sinc; zero outside the signal. For a tone, the error is
    about -90 dB re the signal up to a third of the sampling rate and -78 dB
    at 0.42 of it (20 kHz at 48 kHz); above that the sinc rolls off."""
    margin = 2 * _READ_HALF_WIDTH
    padded = np.concatenate([np.zeros(margin), signal, np.zeros(margin)])
    offsets = np.arange(-_READ_HALF_WIDTH + 1, _READ_HALF_WIDTH + 1)
    out = np.zeros(len(positions))
    inside = np.flatnonzero(
        (positions > -_READ_HALF_WIDTH) & (positions < len(signal) + _READ_HALF_WIDTH - 1)
    )
    for start in range(0, len(inside), chunk):
        rows = inside[start : start + chunk]
        whole = np.floor(positions[rows]).astype(int)
        distance = positions[rows, None] - (whole[:, None] + offsets[None, :])
        window = i0(_READ_KAISER_BETA * np.sqrt(np.clip(1 - (distance / _READ_HALF_WIDTH) ** 2, 0, None)))
        taps = np.sinc(distance) * window / i0(_READ_KAISER_BETA)
        out[rows] = np.sum(padded[whole[:, None] + offsets[None, :] + margin] * taps, axis=1)
    return out


def _emission_times(
    arrival_times: np.ndarray, delay: CubicSpline, duration: float, lookup_times: np.ndarray
) -> np.ndarray:
    """Solve t = t_e + delay(t_e) for t_e at every arrival time t. There is one
    solution for every t only while the delay shrinks by less than a second per
    second, that is, while the source comes toward the ear slower than sound;
    otherwise this raises. Outside the sound the delay is held, so there t_e
    moves one for one with t. The solution is found by inverting t(t_e) on a
    fine grid, then refined by Newton's method."""
    rate = delay.derivative()
    grid = np.linspace(0.0, duration, 8 * len(lookup_times))  # between lookup times too
    arrival_on_grid = grid + delay(grid)
    if np.min(rate(grid)) <= -1 or np.any(np.diff(arrival_on_grid) <= 0):
        raise ValueError(
            "an ear's delay shrinks faster than time passes: the source comes toward it faster than "
            "sound, or nearly as fast where the measured HRIR onsets step between distances"
        )
    emission = np.interp(arrival_times, arrival_on_grid, grid)
    before, after = arrival_times < arrival_on_grid[0], arrival_times > arrival_on_grid[-1]
    emission[before] = arrival_times[before] - delay(0.0)
    emission[after] = arrival_times[after] - delay(duration)
    for _ in range(50):
        clipped = np.clip(emission, 0.0, duration)
        inside = (emission > 0) & (emission < duration)
        slope = 1 + np.where(inside, rate(clipped), 0.0)
        updated = emission - (emission + delay(clipped) - arrival_times) / slope
        change = np.max(np.abs(updated - emission))
        emission = updated
        if change < 1e-12:
            return emission
    raise ValueError("the ear delays change too fast to follow")


def move_sound(
    sound: Sound,
    trajectory,
    hrirs: HRIRSet,
    *,
    room: Sound | None = None,
    drr_db: float = 0.0,
    hop: float = 5e-3,
    speed_of_sound: float = SPEED_OF_SOUND,
) -> Sound:
    """Render a mono sound moving along a path, through measured HRIRs.

    Parameters
    ----------
    trajectory
        Where the source is, as a function of the time it emits:

        - a function ``t -> positions``, Cartesian [m], shape ``(n, 3)``, such
          as :func:`hcc_trajectory` returns;
        - a ``(times, points)`` pair: ``(N,)`` times [s] and ``(N, 3)``
          Cartesian points, interpolated linearly in between and held before
          and after;
        - an ``(N, 3)`` array of points, passed at a constant rate over the
          sound's duration (:func:`linear_trajectory`, :func:`circular_trajectory`).
    hrirs
        The measured HRIRs. Between measured distances they are interpolated;
        beyond them, distance acts through travel time and 1/r only (see the
        module notes). A set with one distance renders every distance that way.
    room
        A reverberant tail, one or two channels, such as
        ``so.synth_ir(rt60, fs, n_channels=2)`` (without ``drr_db``, so it has
        no direct sound). It is driven by the source as it arrives, without the
        1/r, so its level stays put while the direct sound falls with distance,
        and each ear hears it through the HRIRs averaged over directions, since
        reverberation arrives from everywhere.
    drr_db
        Direct-to-reverberant energy ratio at 1 m, averaged over directions.
    hop
        Spacing [s] at which the HRIR shapes are interpolated and switched
        with raised-cosine cross-fades.
    speed_of_sound
        In m/s, for distances beyond the measured ones.

    Each ear reads the sound through its own delay, the HRIR onset at the
    source's position, which changes smoothly from sample to sample, so a
    source changing distance glides in pitch (Doppler) instead of being
    cross-faded between fixed delays, which would comb-filter. The output
    starts when the source starts emitting; each ear hears it a delay later.
    """
    signal = sound.mono().data[:, 0]
    fs = sound.fs
    duration = max(len(signal) - 1, 1) / fs
    path = _as_path(trajectory, duration)

    # Ear delays and the far-field level change, looked up every millisecond
    # and smoothed with a cubic spline so the delay has no kinks.
    lookup_times = np.linspace(0, duration, max(4, int(np.ceil(duration / _DELAY_STEP)) + 1))
    measured_points, radii = hrirs._nearest_measured(path(lookup_times))
    measured_radii = np.linalg.norm(measured_points, axis=1)
    extra_delays = (radii - measured_radii) / speed_of_sound
    ear_delays = hrirs._onset_times(measured_points) + extra_delays[:, None] - hrirs._pre / hrirs.fs
    if np.any(ear_delays < 0):
        raise ValueError("the path comes closer than sound can travel from it; check the distances")
    level = CubicSpline(lookup_times, measured_radii / radii)

    # HRIR shapes every hop; the shapes hold no delay, so switching them is clean.
    n_shapes = max(2, int(np.ceil(duration / hop)) + 1)
    shape_times = np.linspace(0, duration, n_shapes)
    shape_points, _ = hrirs._nearest_measured(path(shape_times))
    shapes, _ = hrirs._shapes(shape_points, fs)
    shape_step = shape_times[1] - shape_times[0]

    n_out = len(signal) + int(np.ceil(ear_delays.max() * fs)) + _READ_HALF_WIDTH + 1
    arrival_times = np.arange(n_out) / fs
    out = np.zeros((n_out + shapes.shape[-1] - 1, 2))
    emission_by_ear = []
    for ear in range(2):
        emission = _emission_times(
            arrival_times, CubicSpline(lookup_times, ear_delays[:, ear]), duration, lookup_times
        )
        emission_by_ear.append(emission)
        arriving = _read_between_samples(signal, emission * fs) * level(np.clip(emission, 0, duration))
        # raised-cosine cross-fades between shapes, in the emission time of each output sample
        position = np.clip(emission / shape_step, 0, n_shapes - 1)
        for shape_index in range(n_shapes):
            first, last = np.searchsorted(position, [shape_index - 1, shape_index + 1], side="right")
            first = max(first - 1, 0)
            offset = position[first:last] - shape_index
            window = np.where(np.abs(offset) < 1, (1 + np.cos(np.pi * offset)) / 2, 0.0)
            if not window.any():
                continue
            segment = fftconvolve(arriving[first:last] * window, shapes[shape_index, ear])
            out[first : first + len(segment), ear] += segment

    if room is not None:
        tail = room.to_channels(2) if room.n_channels == 1 else room
        if tail.fs != fs or tail.n_channels != 2:
            raise ValueError("room must be one or two channels at the sound's sampling rate")
        # Reverberation reaches the ears from every direction, so each ear hears the tail
        # through the HRIRs' average over directions.
        direct_energy, diffuse_filters = hrirs._diffuse_field(fs)
        tail_at_ears = fftconvolve(tail.data, diffuse_filters.T, axes=0)
        tail_energy = np.mean(np.sum(tail_at_ears**2, axis=0))
        scale = np.sqrt(direct_energy * 10 ** (-drr_db / 10) / tail_energy)
        arriving = _read_between_samples(signal, (emission_by_ear[0] + emission_by_ear[1]) / 2 * fs)
        reverberant = fftconvolve(arriving[:, None], tail_at_ears * scale, axes=0)
        total = np.zeros((max(len(out), len(reverberant)), 2))
        total[: len(out)] += out
        total[: len(reverberant)] += reverberant
        out = total
    return Sound(out, fs)
