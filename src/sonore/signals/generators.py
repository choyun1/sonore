"""Stimulus generators.

Every generator returns a :class:`~sonore.Sound` normalized to RMS = 1.
Anything random takes an ``rng`` argument (a seed or ``np.random.Generator``)
so stimuli can be regenerated exactly.
"""

from __future__ import annotations

import numbers
import warnings
from collections.abc import Callable
from typing import Protocol, overload

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import chirp

from sonore.core.sound import Sound
from sonore.core.utils import as_rng, n_samples, time_axis

__all__ = [
    "silence",
    "pure_tone",
    "harmonic_complex",
    "schroeder_complex",
    "square_wave",
    "sawtooth_wave",
    "pulse_train",
    "linear_chirp",
    "exponential_chirp",
    "gaussian_noise",
    "correlated_noise",
    "iterated_ripple_noise",
]

RNG = int | np.random.Generator | None


def _finish(data: np.ndarray, fs: float) -> Sound:
    s = Sound(data, fs)
    return s.normalize() if s.rms > 0 else s


def silence(duration: float, fs: float, n_channels: int = 1) -> Sound:
    return Sound(np.zeros((n_samples(duration, fs), n_channels)), fs)


def pure_tone(duration: float, fs: float, freq: float, phase: float = 0.0) -> Sound:
    """``cos(2*pi*freq*t + phase)``."""
    t = time_axis(n_samples(duration, fs), fs)
    return _finish(np.cos(2 * np.pi * freq * t + phase), fs)


# ----------------------------------------------------------- harmonic sounds
_PHASE_PRESETS = ("cosine", "sine", "alternating", "random", "schroeder+", "schroeder-")


class F0Contour(Protocol):
    """Anything with frame times ``t`` [s] and F0 values ``f0`` [Hz], 0 where
    unvoiced, such as an :class:`~sonore.analysis.f0.F0Track`. ``f0`` is one
    row of values per channel, or a single row."""

    t: ArrayLike
    f0: ArrayLike


Amplitudes = ArrayLike | Callable[[np.ndarray, np.ndarray], ArrayLike] | None


def _harmonic_phases(harmonics: np.ndarray, phases, rng: RNG) -> np.ndarray:
    if not isinstance(phases, str):
        phases = np.asarray(phases, float)
        if phases.ndim > 0 and phases.shape != harmonics.shape:
            raise ValueError(
                f"phases has {phases.size} values; this sound has {harmonics.size} harmonics "
                "and needs one starting phase for each"
            )
        return np.broadcast_to(phases, harmonics.shape)
    n = harmonics
    N = len(harmonics)
    match phases:
        case "cosine":
            return np.zeros(N)
        case "sine":
            return np.full(N, -np.pi / 2)
        case "alternating":
            return np.where(n % 2 == 1, 0.0, -np.pi / 2)
        case "random":
            return as_rng(rng).uniform(-np.pi, np.pi, N)
        case "schroeder+" | "schroeder-":
            sign = 1 if phases.endswith("+") else -1
            return sign * np.pi * n * (n + 1) / N
    raise ValueError(f"phases must be an array or one of {_PHASE_PRESETS}")


def _contour(f0) -> tuple[np.ndarray, np.ndarray] | None:
    """``(frame times, values of shape (n_channels, n_frames))`` for an F0
    contour, or None for a fixed F0 (a number)."""
    match f0:
        case numbers.Real():
            return None
        case (t, values):
            pass
        case object(t=t, f0=values):
            pass
        case _:
            raise TypeError(
                "f0 must be a number, a (times, values) pair, or an object with .t and .f0 such as an F0Track"
            )
    t = np.asarray(t, float)
    values = np.atleast_2d(np.asarray(values, float))
    if t.ndim != 1 or len(t) == 0 or values.ndim != 2 or values.shape[1] != len(t):
        raise ValueError(
            f"an F0 contour needs one value per frame time: got {np.shape(t)} times and {values.shape} values"
        )
    if np.any(np.diff(t) <= 0):
        raise ValueError("an F0 contour's frame times must increase")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("an F0 contour's values must be finite and >= 0 (0 where unvoiced)")
    return t, values


def _max_harmonic(values: np.ndarray, f_max: float) -> int:
    """The highest harmonic number that is below ``f_max`` somewhere on the contour."""
    voiced = values[values > 0]
    return int(f_max // voiced.min()) if voiced.size else 0


def _n_max(f0, fs: float, f_max: float | None) -> int:
    """The highest harmonic number a harmonic waveform on ``f0`` can hold:
    below Nyquist for a fixed F0, below ``f_max`` somewhere for a contour."""
    contour = _contour(f0)
    if contour is None:
        return int(np.ceil(fs / 2 / f0)) - 1
    return _max_harmonic(contour[1], 0.45 * fs if f_max is None else f_max)


def _taper(freq: np.ndarray, f_max: float) -> np.ndarray:
    """1 below 0.9 f_max, falling as cos^2 to 0 at f_max and above."""
    lo = 0.9 * f_max
    u = np.clip((freq - lo) / (f_max - lo), 0.0, 1.0)
    return np.cos(np.pi / 2 * u) ** 2


def _voicing_gate(
    t: np.ndarray, t_frames: np.ndarray, voiced: np.ndarray, ramp: float, fs: float
) -> np.ndarray:
    """1 where the nearest frame is voiced, 0 elsewhere, every step smoothed by
    a Hann window ``ramp`` seconds long. The ends are extended, not faded."""
    gate = (np.interp(t, t_frames, voiced.astype(float)) >= 0.5).astype(float)
    m = int(round(ramp * fs))
    if m < 2:
        return gate
    w = np.hanning(m + 2)[1:-1]
    padded = np.pad(gate, m, mode="edge")
    return np.convolve(padded, w / w.sum(), mode="same")[m:-m]


def _gains(amplitudes: Amplitudes, harmonics: np.ndarray) -> np.ndarray | None:
    """Per-harmonic amplitudes as an array, or None when they are a function."""
    if callable(amplitudes):
        return None
    if amplitudes is None:
        return np.ones(len(harmonics))
    amplitudes = np.asarray(amplitudes, float)
    if amplitudes.shape != harmonics.shape:
        raise ValueError("amplitudes must match harmonics")
    return amplitudes


@overload
def harmonic_complex(
    duration: float,
    fs: float,
    f0: float,
    harmonics: ArrayLike | None = None,
    amplitudes: Amplitudes = None,
    phases: ArrayLike | str = "cosine",
    rng: RNG = None,
) -> Sound: ...


@overload
def harmonic_complex(
    duration: float,
    fs: float,
    f0: tuple[ArrayLike, ArrayLike] | F0Contour,
    harmonics: ArrayLike | None = None,
    amplitudes: Amplitudes = None,
    phases: ArrayLike | str = "cosine",
    rng: RNG = None,
    *,
    f_max: float | None = None,
    ramp: float = 0.005,
    unvoiced: str = "silence",
) -> Sound: ...


def harmonic_complex(
    duration,
    fs,
    f0,
    harmonics=None,
    amplitudes=None,
    phases="cosine",
    rng=None,
    *,
    f_max=None,
    ramp=0.005,
    unvoiced="silence",
):
    """Sum of harmonics ``a_n cos(n * Phi(t) + phi_n)`` of a fixed or moving F0.

    ``f0`` is either a number, for ``Phi = 2*pi*f0*t``, or an F0 contour:
    a ``(times, values)`` pair or anything with ``.t`` and ``.f0``, such as
    an :class:`~sonore.analysis.f0.F0Track`, with values in Hz and 0 where
    unvoiced (one row per channel for a multichannel sound).

    ``harmonics`` are the harmonic numbers; by default every one below
    Nyquist (fixed F0) or below ``f_max`` (contour). ``amplitudes`` is one
    value per harmonic, or a function ``amplitudes(t, f)`` giving the gain
    at times ``t`` and frequencies ``f`` (a spectral envelope, evaluated at
    every harmonic's frequency). ``phases`` are starting phases: an array,
    one per harmonic, or one of ``"cosine"``, ``"sine"``, ``"alternating"``,
    ``"random"``, ``"schroeder+"``, ``"schroeder-"``.

    A fixed F0 drops harmonics at or above Nyquist, with a warning.

    A contour (``docs/design/harmonic-source.md``) is filled across its
    unvoiced gaps and interpolated linearly to the sample rate, and the phase
    is its exact running integral, so every harmonic follows ``n`` times the
    contour with no jumps. Each harmonic fades out (``cos^2``) between
    ``0.9 * f_max`` and ``f_max`` (default ``0.45 * fs``), so nothing aliases
    as the pitch rises. Harmonics are switched off where the contour is
    unvoiced, with Hann ramps ``ramp`` seconds long; ``unvoiced="noise"``
    fills those stretches with white noise of the harmonics' power instead
    of silence. Values are held beyond the first and last frames. With a
    contour, ``phases`` has one value per harmonic number up to the highest
    that fits below ``f_max`` at the lowest voiced F0, unless ``harmonics``
    is given.
    """
    contour = _contour(f0)
    if contour is not None:
        return _contour_complex(
            duration, fs, *contour, harmonics, amplitudes, phases, rng, f_max, ramp, unvoiced
        )
    if harmonics is None:
        if f0 <= 0:
            raise ValueError(f"f0 must be positive, got {f0:g}")
        harmonics = np.arange(1, int(np.ceil(fs / 2 / f0)))
    harmonics = np.atleast_1d(np.asarray(harmonics))
    gains = _gains(amplitudes, harmonics)
    phi = _harmonic_phases(harmonics, phases, rng)

    keep = harmonics * f0 < fs / 2
    if not keep.all():
        warnings.warn(f"dropping {np.sum(~keep)} harmonic(s) at or above Nyquist", stacklevel=2)
    t = time_axis(n_samples(duration, fs), fs)
    data = np.zeros_like(t)
    if gains is None:
        gains = [amplitudes(t, np.full_like(t, n * f0)) for n in harmonics[keep]]
    else:
        gains = gains[keep]
    for n, a, p in zip(harmonics[keep], gains, phi[keep], strict=True):
        data += a * np.cos(2 * np.pi * n * f0 * t + p)
    return _finish(data, fs)


def _contour_complex(
    duration, fs, t_frames, values, harmonics, amplitudes, phases, rng, f_max, ramp, unvoiced
) -> Sound:
    f_max = 0.45 * fs if f_max is None else float(f_max)
    if not 0 < f_max <= fs / 2:
        raise ValueError(f"f_max must be in (0, fs/2], got {f_max:g}")
    if unvoiced not in ("silence", "noise"):
        raise ValueError(f"unvoiced must be 'silence' or 'noise', got {unvoiced!r}")
    rng = as_rng(rng)
    if harmonics is None:
        harmonics = np.arange(1, _max_harmonic(values, f_max) + 1)
    harmonics = np.atleast_1d(np.asarray(harmonics))
    gains = _gains(amplitudes, harmonics)
    phi = _harmonic_phases(harmonics, phases, rng)

    n = n_samples(duration, fs)
    t = time_axis(n, fs)
    data = np.zeros((n, len(values)))
    for c, v in enumerate(values):
        voiced = v > 0
        harm = np.zeros(n)
        gate = np.zeros(n)
        if voiced.any():
            i = np.arange(len(v))
            f = np.interp(t, t_frames, np.interp(i, i[voiced], v[voiced]))
            # the trapezoid rule: exact for f linear between samples
            phase = 2 * np.pi / fs * np.concatenate([[0.0], np.cumsum((f[1:] + f[:-1]) / 2)])
            for k, (n_k, p) in enumerate(zip(harmonics, phi, strict=True)):
                freq = n_k * f
                a = amplitudes(t, freq) if gains is None else gains[k]
                harm += a * _taper(freq, f_max) * np.cos(n_k * phase + p)
            gate = _voicing_gate(t, t_frames, voiced, ramp, fs)
        data[:, c] = gate * harm
        if unvoiced == "noise":
            weight = gate.sum()
            power = np.sum(gate * harm**2) / weight if weight > 0 else 1.0
            data[:, c] += (1 - gate) * np.sqrt(power) * rng.standard_normal(n)
    return _finish(data, fs)


def schroeder_complex(
    duration: float,
    fs: float,
    f0: float | tuple[ArrayLike, ArrayLike] | F0Contour,
    n_harmonics: int | None = None,
    sign: int = 1,
    **contour,
) -> Sound:
    """Equal-amplitude harmonic complex with Schroeder phases
    ``sign * pi * n(n+1)/N``. ``n_harmonics`` defaults to all below Nyquist
    (or below ``f_max`` for an F0 contour). ``f0`` and the keyword arguments
    for a contour are as in :func:`harmonic_complex`."""
    n_max = _n_max(f0, fs, contour.get("f_max"))
    N = n_max if n_harmonics is None else min(n_harmonics, n_max)
    return harmonic_complex(
        duration, fs, f0, np.arange(1, N + 1), phases="schroeder+" if sign > 0 else "schroeder-", **contour
    )


def _waveform(duration, fs, f0, phase, harmonics, amplitudes, kind, contour) -> Sound:
    """A named band-limited waveform as a harmonic complex: harmonic ``n`` has
    amplitude ``amplitudes[i]`` and phase ``n * phase``, as a sine or a cosine."""
    phases = harmonics * phase - (np.pi / 2 if kind == "sin" else 0.0)
    return harmonic_complex(duration, fs, f0, harmonics, amplitudes, phases, **contour)


def _fixed_only(f0, name: str) -> None:
    if _contour(f0) is not None:
        raise ValueError(
            f"a {name} that is not band-limited needs a fixed f0; use bandlimited=True for a contour"
        )


def square_wave(
    duration: float,
    fs: float,
    f0: float | tuple[ArrayLike, ArrayLike] | F0Contour,
    phase: float = 0.0,
    bandlimited: bool = True,
    **contour,
) -> Sound:
    """Square wave. Band-limited (odd harmonics below Nyquist) unless
    ``bandlimited=False``, which aliases badly at high f0. A band-limited
    square wave also takes an F0 contour, as :func:`harmonic_complex` does."""
    if not bandlimited:
        _fixed_only(f0, "square wave")
        t = time_axis(n_samples(duration, fs), fs)
        return _finish(np.sign(np.sin(2 * np.pi * f0 * t + phase)), fs)
    n = np.arange(1, _n_max(f0, fs, contour.get("f_max")) + 1, 2)
    return _waveform(duration, fs, f0, phase, n, 1 / n, "sin", contour)


def sawtooth_wave(
    duration: float,
    fs: float,
    f0: float | tuple[ArrayLike, ArrayLike] | F0Contour,
    phase: float = 0.0,
    bandlimited: bool = True,
    **contour,
) -> Sound:
    """Rising sawtooth. Band-limited unless ``bandlimited=False``. A
    band-limited sawtooth also takes an F0 contour, as
    :func:`harmonic_complex` does."""
    if not bandlimited:
        from scipy.signal import sawtooth

        _fixed_only(f0, "sawtooth")
        t = time_axis(n_samples(duration, fs), fs)
        return _finish(sawtooth(2 * np.pi * f0 * t + phase + np.pi), fs)
    n = np.arange(1, _n_max(f0, fs, contour.get("f_max")) + 1)
    return _waveform(duration, fs, f0, phase, n, (-1.0) ** (n + 1) / n, "sin", contour)


def pulse_train(
    duration: float,
    fs: float,
    f0: float | tuple[ArrayLike, ArrayLike] | F0Contour,
    phase: float = 0.0,
    bandlimited: bool = True,
    **contour,
) -> Sound:
    """Periodic pulse (click) train.

    Band-limited: equal-amplitude cosine-phase harmonics up to Nyquist; this
    also takes an F0 contour, as :func:`harmonic_complex` does. Otherwise
    single-sample impulses at the nearest sample to each period.
    ``phase`` (radians of the fundamental) shifts the pulses in time.
    """
    if bandlimited:
        n = np.arange(1, _n_max(f0, fs, contour.get("f_max")) + 1)
        return _waveform(duration, fs, f0, phase, n, np.ones(len(n)), "cos", contour)
    _fixed_only(f0, "pulse train")
    N = n_samples(duration, fs)
    times = (np.arange(0, duration * f0 + 1) - phase / (2 * np.pi)) / f0
    idx = np.round(times * fs).astype(int)
    data = np.zeros(N)
    data[idx[(idx >= 0) & (idx < N)]] = 1.0
    return _finish(data, fs)


def linear_chirp(duration: float, fs: float, f0: float, f1: float, phase: float = 0.0) -> Sound:
    """Linear frequency sweep from ``f0`` to ``f1``."""
    t = time_axis(n_samples(duration, fs), fs)
    return _finish(chirp(t, f0, duration, f1, "linear", phi=np.degrees(phase)), fs)


def exponential_chirp(duration: float, fs: float, f0: float, f1: float, phase: float = 0.0) -> Sound:
    """Exponential (log-frequency) sweep from ``f0`` to ``f1``."""
    t = time_axis(n_samples(duration, fs), fs)
    return _finish(chirp(t, f0, duration, f1, "logarithmic", phi=np.degrees(phase)), fs)


# ------------------------------------------------------------------- noises
SpectrumLike = Callable[[np.ndarray], np.ndarray] | tuple[ArrayLike, ArrayLike]


def _spectral_gain(
    f: np.ndarray,
    band: tuple[float, float] | None,
    tilt: float,
    spectrum,
    tilt_ref: float,
) -> np.ndarray:
    gain = np.ones_like(f)
    if band is not None:
        lo, hi = band
        gain[(f < lo) | (f > hi)] = 0.0
    if tilt:
        with np.errstate(divide="ignore"):
            gain *= np.where(f > 0, (f / tilt_ref) ** (tilt / (20 * np.log10(2))), 0.0)
    if spectrum is not None:
        if hasattr(spectrum, "level_at"):  # a sonore Spectrum
            db = spectrum.level_at(f)
        elif callable(spectrum):
            db = spectrum(f)
        else:
            sf, sdb = map(np.asarray, spectrum)
            db = np.interp(f, sf, sdb)
        gain *= 10 ** (np.asarray(db) / 20)
    return gain


def gaussian_noise(
    duration: float,
    fs: float,
    band: tuple[float, float] | None = None,
    tilt: float = 0.0,
    spectrum: SpectrumLike | None = None,
    n_channels: int = 1,
    rng: RNG = None,
    tilt_ref: float = 1000.0,
) -> Sound:
    """Gaussian noise, optionally spectrally shaped.

    Parameters
    ----------
    band
        ``(f_lo, f_hi)`` passband in Hz (brick-wall).
    tilt
        Spectral slope in dB/octave (``-3`` = pink, ``-6`` = brown).
    spectrum
        Target spectrum level in dB: a :class:`~sonore.Spectrum`, a
        function ``f -> dB``, or a ``(freqs, dB)`` pair to interpolate.
    n_channels
        Independent noise in each channel.
    """
    rng = as_rng(rng)
    N = n_samples(duration, fs)
    f = np.fft.rfftfreq(N, 1 / fs)
    gain = _spectral_gain(f, band, tilt, spectrum, tilt_ref)
    spec = np.fft.rfft(rng.standard_normal((N, n_channels)), axis=0) * gain[:, None]
    data = np.fft.irfft(spec, n=N, axis=0)
    data -= data.mean(axis=0)
    return _finish(data, fs)


def correlated_noise(duration: float, fs: float, corr: float = 1.0, rng: RNG = None, **noise_kwargs) -> Sound:
    """Two-channel noise with interaural correlation ``corr`` (in [-1, 1]).
    Extra keyword arguments go to :func:`gaussian_noise`."""
    if not -1 <= corr <= 1:
        raise ValueError("corr must be in [-1, 1]")
    n = gaussian_noise(duration, fs, n_channels=2, rng=rng, **noise_kwargs).data
    a, b = np.sqrt((1 + corr) / 2), np.sqrt((1 - corr) / 2)
    return Sound(np.column_stack([a * n[:, 0] + b * n[:, 1], a * n[:, 0] - b * n[:, 1]]), fs)


def iterated_ripple_noise(
    duration: float,
    fs: float,
    delay: float,
    gain: float = 1.0,
    iterations: int = 16,
    network: str = "add-same",
    rng: RNG = None,
    **noise_kwargs,
) -> Sound:
    """Iterated rippled noise (Yost, 1996). Pitch is at ``1/delay``.

    ``network="add-same"`` (IRNS) iterates ``y <- y + g*delay(y)``;
    ``"add-original"`` (IRNO) iterates ``y <- x + g*delay(y)``. Implemented
    exactly in the frequency domain (fractional delays are fine), with a
    warm-up segment discarded so the output is stationary.
    """
    warmup = int(np.ceil(iterations * delay * fs))
    N = n_samples(duration, fs)
    x = gaussian_noise((N + warmup) / fs, fs, rng=rng, **noise_kwargs).data[:, 0]
    f = np.fft.rfftfreq(len(x), 1 / fs)
    z = gain * np.exp(-2j * np.pi * f * delay)
    if network == "add-same":
        H = (1 + z) ** iterations
    elif network == "add-original":
        H = sum(z**i for i in range(iterations + 1))
    else:
        raise ValueError("network must be 'add-same' or 'add-original'")
    y = np.fft.irfft(np.fft.rfft(x) * H, n=len(x))[warmup:]
    return _finish(y, fs)
