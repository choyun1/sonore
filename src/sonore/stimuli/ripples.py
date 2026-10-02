"""Spectrotemporal ripples: sounds defined by their modulation content.

A *pattern* is an envelope over time and log-frequency, ``E(t, x)``, where
``x`` is octaves above ``f_lo``. :func:`ripple_sound` imposes a pattern on a
*carrier*, which supplies the fine structure.

Patterns
--------
:class:`Ripple`
    A moving ripple (Kowalski, Depireux & Shamma, 1996; Chi et al., 1999):
    ``1 + depth * sin(2*pi*(rate*t + density*x) + phase)``. ``rate`` is in Hz
    and ``density`` in cycles/octave. With positive rate and density the
    ripple drifts *downward* in frequency; a negative rate drifts upward.
    Ripples add: ``Ripple(4, 1) + Ripple(-8, 2)`` is a :class:`RippleSum`.
:class:`DynamicRipple`
    A dynamic moving ripple (Escabí & Schreiner, 2002) whose rate and density
    wander slowly and randomly within given ranges, for STRF estimation.
Any callable ``f(t, x)``
    Evaluated with ``t`` as a row and ``x`` as a column, so ordinary numpy
    broadcasting works, e.g. ``lambda t, x: 1 + 0.5 * np.sin(2*np.pi*(3*t + x**2))``.

Carriers
--------
``"tones"``: log-spaced tones with random phases, the classic ripple carrier.
``"harmonic"``: harmonics of ``f0``. ``"noise"``: narrowband Gaussian noise
per channel. ``"low-noise"``: the fine structure of that noise with its
envelope flattened, so it adds no envelope fluctuations of its own. Or any
:class:`~sonore.Sound`, whose fine structure is used the same way.

All carriers are scaled to equal energy per octave, so switching carriers
changes the fine structure but not the long-term spectrum.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.special import ndtr

from sonore.analysis.envelopes import Envelopes
from sonore.analysis.filterbank import OctaveFilterbank, Subbands
from sonore.core.sound import Sound
from sonore.core.utils import as_rng, n_samples, time_axis
from sonore.signals.generators import gaussian_noise

__all__ = ["Ripple", "RippleSum", "DynamicRipple", "ripple_sound", "render"]

_SCALES = ("linear", "db")


class _Pattern:
    """Shared behaviour: addition into sums, and plotting."""

    def __add__(self, other):
        if isinstance(other, int | float) and other == 0:  # lets sum() work
            return self
        if isinstance(other, Ripple | RippleSum) and isinstance(self, Ripple | RippleSum):
            return RippleSum(_components(self) + _components(other))
        return NotImplemented

    __radd__ = __add__

    def render(self, filterbank: OctaveFilterbank, duration: float, fs: float) -> Envelopes:
        """The pattern evaluated at each band center of ``filterbank`` (with
        ``x`` in octaves above ``filterbank.f_lo``), as Envelopes."""
        return render(self, filterbank, duration, fs)

    def plot(self, duration: float = 1.0, f_lo: float = 250.0, f_hi: float = 8000.0, ax=None, **kwargs):
        """Show the envelope pattern (in dB) over time and frequency."""
        from sonore.plotting import plot_ripple_pattern

        return plot_ripple_pattern(self, duration, f_lo, f_hi, ax=ax, **kwargs)


def _components(pattern) -> tuple[Ripple, ...]:
    return (pattern,) if isinstance(pattern, Ripple) else pattern.components


@dataclass(frozen=True)
class Ripple(_Pattern):
    """A moving ripple ``1 + depth*sin(2*pi*(rate*t + density*x) + phase)``.

    Parameters
    ----------
    rate
        Temporal modulation [Hz]. Positive: drifts down in frequency.
    density
        Spectral modulation [cycles/octave].
    depth
        ``scale="linear"``: modulation depth in [0, 1].
        ``scale="db"``: peak-to-peak depth in dB (the envelope is
        ``10**((depth/2)*sin(...)/20)``).
    phase
        Starting phase [radians].
    """

    rate: float
    density: float
    depth: float = 0.9
    phase: float = 0.0
    scale: str = "linear"

    def __post_init__(self):
        if self.scale not in _SCALES:
            raise ValueError(f"scale must be one of {_SCALES}")
        if self.scale == "linear" and not 0 <= self.depth <= 1:
            raise ValueError("linear depth must be in [0, 1]; use scale='db' for dB depths")

    def __repr__(self) -> str:
        depth = f"{self.depth:g}" if self.scale == "linear" else f"{self.depth:g} dB"
        phase = f", phase {self.phase:g}" if self.phase else ""
        return f"Ripple({self.rate:g} Hz, {self.density:g} cyc/oct, depth {depth}{phase})"

    @property
    def direction(self) -> str:
        if self.rate == 0 or self.density == 0:
            return "static" if self.rate == 0 else "temporal only"
        return "downward" if self.rate * self.density > 0 else "upward"

    def envelope(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        return RippleSum((self,)).envelope(t, x)


@dataclass(frozen=True)
class RippleSum(_Pattern):
    """A sum of ripples sharing one depth scale. Linear depths must total <= 1
    so the envelope stays non-negative."""

    components: tuple[Ripple, ...]

    def __post_init__(self):
        scales = {ripple.scale for ripple in self.components}
        if len(scales) > 1:
            raise ValueError("cannot add linear-scale and dB-scale ripples")
        if scales == {"linear"} and sum(ripple.depth for ripple in self.components) > 1 + 1e-12:
            raise ValueError("linear ripple depths sum to more than 1; the envelope would go negative")

    def __repr__(self) -> str:
        return " + ".join(repr(ripple) for ripple in self.components)

    @property
    def scale(self) -> str:
        return self.components[0].scale

    def envelope(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        t, x = np.asarray(t, float), np.asarray(x, float)
        total = np.zeros((len(x), len(t)))
        for ripple in self.components:
            modulation = np.sin(
                2 * np.pi * (ripple.rate * t[None, :] + ripple.density * x[:, None]) + ripple.phase
            )
            total += (ripple.depth if self.scale == "linear" else ripple.depth / 2) * modulation
        return 1 + total if self.scale == "linear" else 10 ** (total / 20)


@dataclass(frozen=True)
class DynamicRipple(_Pattern):
    """Dynamic moving ripple (Escabí & Schreiner, 2002).

    The envelope is ``10**((depth/2) * sin(2*pi*density(t)*x + Phi(t)) / 20)``
    with ``Phi(t) = 2*pi * integral of rate(t)``. ``rate(t)`` and
    ``density(t)`` are independent, slowly varying random processes, uniformly
    distributed over their ranges, whose fastest changes are limited to
    ``rate_change`` and ``density_change`` Hz.

    Note that ``rate(t)`` is the temporal modulation at ``x = 0`` (``f_lo``).
    Elsewhere the local rate is ``rate(t) + x * d(density)/dt``, because a
    changing density fans the ripple out across frequency. With fast density
    changes, a good share of the modulation energy lies outside
    ``rate_range`` (about 40% for the defaults over 5 octaves, under 10% with
    ``density_change=0.25``). Analyze with
    ``ModulationSpectrum.octave(..., scale="db")``, since the pattern is
    defined in dB.

    The defaults follow the ranges commonly used after Escabí & Schreiner
    (2002); check them against the study you are matching.

    The pattern is fully determined by ``seed`` (drawn at random if not
    given, and stored), and it doesn't depend on the sampling rate.
    """

    rate_range: tuple[float, float] = (-350.0, 350.0)
    density_range: tuple[float, float] = (0.0, 4.0)
    rate_change: float = 3.0
    density_change: float = 6.0
    depth: float = 45.0
    seed: int | None = None
    grid_fs: float = field(default=1000.0, repr=False)

    def __post_init__(self):
        if self.seed is None:
            object.__setattr__(self, "seed", int(np.random.default_rng().integers(2**32)))

    @property
    def scale(self) -> str:
        return "db"

    def trajectories(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``rate(t)`` [Hz] and ``density(t)`` [cycles/octave] at times ``t``."""
        t = np.asarray(t, float)
        n_grid = int(np.ceil(t[-1] * self.grid_fs)) + 2
        grid = np.arange(n_grid) / self.grid_fs
        rng = as_rng(self.seed)
        out = []
        for (lo, hi), cutoff in (
            (self.rate_range, self.rate_change),
            (self.density_range, self.density_change),
        ):
            noise = rng.standard_normal(n_grid + 2 * int(self.grid_fs))  # extra samples avoid edge effects
            sos = butter(4, cutoff, fs=self.grid_fs, output="sos")
            noise = sosfiltfilt(sos, noise)[int(self.grid_fs) : int(self.grid_fs) + n_grid]
            uniform = ndtr((noise - noise.mean()) / noise.std())  # Gaussian -> uniform on (0, 1)
            out.append(np.interp(t, grid, lo + (hi - lo) * uniform))
        return out[0], out[1]

    def envelope(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        t, x = np.asarray(t, float), np.asarray(x, float)
        rate, density = self.trajectories(t)
        dt = t[1] - t[0] if len(t) > 1 else 1.0
        phi = 2 * np.pi * np.cumsum(rate) * dt
        level_db = (self.depth / 2) * np.sin(2 * np.pi * density[None, :] * x[:, None] + phi[None, :])
        return 10 ** (level_db / 20)


Pattern = Ripple | RippleSum | DynamicRipple | Callable[[np.ndarray, np.ndarray], np.ndarray]


def _evaluate(pattern: Pattern, t: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Envelope with shape ``(len(x), len(t))``."""
    if hasattr(pattern, "envelope"):
        envelope = pattern.envelope(t, x)
    else:
        envelope = np.broadcast_to(pattern(t[None, :], x[:, None]), (len(x), len(t)))
    if np.any(envelope < 0):
        raise ValueError("the envelope pattern must be non-negative")
    return envelope


def _max_density(pattern: Pattern) -> float | None:
    if isinstance(pattern, Ripple | RippleSum):
        return max(abs(ripple.density) for ripple in _components(pattern))
    if isinstance(pattern, DynamicRipple):
        return max(abs(bound) for bound in pattern.density_range)
    return None


def _max_rate(pattern: Pattern) -> float | None:
    if isinstance(pattern, Ripple | RippleSum):
        return max(abs(ripple.rate) for ripple in _components(pattern))
    if isinstance(pattern, DynamicRipple):
        return max(abs(bound) for bound in pattern.rate_range)
    return None


def _check_resolution(pattern: Pattern, per_octave: float, what: str) -> None:
    density = _max_density(pattern)
    if density is not None and density > per_octave / 2:
        warnings.warn(
            f"ripple density {density:g} cyc/oct exceeds the spectral Nyquist limit of the "
            f"{what} ({per_octave / 2:.3g} cyc/oct); the pattern will alias there",
            stacklevel=3,
        )


def render(pattern: Pattern, filterbank: OctaveFilterbank, duration: float, fs: float) -> Envelopes:
    """Evaluate any pattern (including a plain ``f(t, x)``) on the band centers
    of an octave filterbank, giving :class:`~sonore.analysis.envelopes.Envelopes`.
    Compare it with a sound's measured envelopes on the same filterbank."""
    if not isinstance(filterbank, OctaveFilterbank):
        raise TypeError("patterns are defined in octaves; use an OctaveFilterbank")
    t = time_axis(n_samples(duration, fs), fs)
    x = np.log2(filterbank.cfs / filterbank.f_lo)
    return Envelopes(_evaluate(pattern, t, x).T, fs, filterbank)


def _flat_noise_bands(noise: Sound, filterbank: OctaveFilterbank) -> Subbands:
    """Noise bands shaped by the *squared* filter responses and scaled to equal
    RMS. Because the squared responses sum to 1, these bands add up to a flat
    spectrum without re-filtering, so modulation sidebands survive intact
    (re-filtering with :meth:`Subbands.synthesize` would attenuate fast
    modulations in narrow low-frequency bands)."""
    length = len(noise)
    response = filterbank.rfft_response(length, noise.fs)
    bands = np.fft.irfft(np.fft.rfft(noise.data[:, 0])[:, None] * response**2, n=length, axis=0)
    bands /= np.sqrt(np.mean(bands**2, axis=0, keepdims=True)) + 1e-30
    return Subbands(bands[:, :, None], noise.fs, filterbank)


def ripple_sound(
    pattern: Pattern,
    duration: float,
    fs: float,
    f_lo: float = 250.0,
    f_hi: float = 8000.0,
    carrier: str | Sound = "tones",
    tones_per_octave: float = 20.0,
    f0: float = 100.0,
    bands_per_octave: float = 24.0,
    rng=None,
    chunk: int = 32,
) -> Sound:
    """Synthesize a sound whose spectrotemporal envelope is ``pattern``.

    Parameters
    ----------
    pattern
        A :class:`Ripple`, :class:`RippleSum`, :class:`DynamicRipple`, or a
        function ``f(t, x)`` of time [s] and octaves above ``f_lo``.
    f_lo, f_hi
        Frequency range [Hz]; ``x`` runs from 0 to ``log2(f_hi/f_lo)``.
    carrier
        ``"tones"``, ``"harmonic"``, ``"noise"``, ``"low-noise"``, or a Sound.
    tones_per_octave
        Density of the tone carrier. Spectral modulation up to half this
        (cycles/octave) is representable.
    f0
        Fundamental of the harmonic carrier. Harmonic spacing in octaves is
        coarse at low harmonic numbers, which limits the representable ripple
        density there; a warning is issued when it's exceeded.
    bands_per_octave
        Channel density for noise and Sound carriers.

    The result has RMS = 1. Phases (tones, harmonics) and noise are drawn from
    ``rng``.
    """
    rng = as_rng(rng)
    length = n_samples(duration, fs)
    t = time_axis(length, fs)
    if f_hi >= fs / 2:
        raise ValueError("f_hi must be below Nyquist")
    out = np.zeros(length)

    if isinstance(carrier, str) and carrier in ("tones", "harmonic"):
        if carrier == "tones":
            _check_resolution(pattern, tones_per_octave, "tone carrier")
            tone_index = np.arange(int(np.floor(tones_per_octave * np.log2(f_hi / f_lo))) + 1)
            freqs = f_lo * 2 ** (tone_index / tones_per_octave)
            weights = np.ones(len(freqs))
        else:
            harmonics = np.arange(int(np.ceil(f_lo / f0)), int(np.floor(f_hi / f0)) + 1)
            if len(harmonics) == 0:
                raise ValueError("no harmonics of f0 fall between f_lo and f_hi")
            freqs = harmonics * f0
            weights = 1 / np.sqrt(harmonics)  # equal energy per octave
            _check_resolution(
                pattern,
                1 / np.log2((harmonics[0] + 1) / harmonics[0]),
                f"harmonic carrier's lowest harmonics (f0={f0:g} Hz, near {harmonics[0] * f0:g} Hz)",
            )
        x = np.log2(freqs / f_lo)
        phases = rng.uniform(0, 2 * np.pi, len(freqs))
        for start in range(0, len(freqs), chunk):
            batch = slice(start, start + chunk)
            envelope = _evaluate(pattern, t, x[batch])
            out += np.sum(
                weights[batch, None]
                * envelope
                * np.sin(2 * np.pi * freqs[batch, None] * t[None, :] + phases[batch, None]),
                axis=0,
            )
        return Sound(out, fs).normalize()

    # channel carriers: pattern envelopes x the carrier's band fine structure
    _check_resolution(pattern, bands_per_octave, "channel carrier")
    filterbank = OctaveFilterbank.per_octave(bands_per_octave, f_lo, f_hi)
    if isinstance(carrier, Sound):
        if carrier.fs != fs or len(carrier) < length:
            raise ValueError("carrier sound must have the same fs and be at least as long")
        fine = filterbank.analyze(Sound(carrier.mono().data[:length], fs)).tfs()
    elif carrier == "low-noise":
        # our own noise is periodic, so circular analysis (pad=0) is exact here
        fine = filterbank.analyze(gaussian_noise(duration, fs, rng=rng), pad=0).tfs()
    elif carrier == "noise":
        fine = _flat_noise_bands(gaussian_noise(duration, fs, rng=rng), filterbank)
    else:
        raise ValueError("carrier must be 'tones', 'harmonic', 'noise', 'low-noise', or a Sound")
    envelopes = render(pattern, filterbank, duration, fs).without_edges()  # edges lie outside f_lo..f_hi
    return (envelopes * fine).sum().normalize()
