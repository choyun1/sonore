"""Stimulus generators.

Every generator returns a :class:`~sigtools.Sound` normalized to RMS = 1.
Anything random takes an ``rng`` argument (a seed or ``np.random.Generator``)
so stimuli can be regenerated exactly.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import chirp

from sigtools.sound import Sound
from sigtools.utils import as_rng, n_samples, time_axis

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


def _harmonic_phases(harmonics: np.ndarray, phases, rng: RNG) -> np.ndarray:
    if not isinstance(phases, str):
        return np.broadcast_to(np.asarray(phases, float), harmonics.shape)
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


def harmonic_complex(
    duration: float,
    fs: float,
    f0: float,
    harmonics: ArrayLike,
    amplitudes: ArrayLike | None = None,
    phases: ArrayLike | str = "cosine",
    rng: RNG = None,
) -> Sound:
    """Sum of cosines ``a_n cos(2*pi*n*f0*t + phi_n)``.

    ``phases`` may be an array or one of ``"cosine"``, ``"sine"``,
    ``"alternating"``, ``"random"``, ``"schroeder+"``, ``"schroeder-"``.
    Harmonics at or above Nyquist are dropped with a warning.
    """
    harmonics = np.atleast_1d(np.asarray(harmonics))
    amplitudes = np.ones(len(harmonics)) if amplitudes is None else np.asarray(amplitudes, float)
    if amplitudes.shape != harmonics.shape:
        raise ValueError("amplitudes must match harmonics")
    phi = _harmonic_phases(harmonics, phases, rng)

    keep = harmonics * f0 < fs / 2
    if not keep.all():
        warnings.warn(f"dropping {np.sum(~keep)} harmonic(s) at or above Nyquist", stacklevel=2)
    t = time_axis(n_samples(duration, fs), fs)
    data = np.zeros_like(t)
    for n, a, p in zip(harmonics[keep], amplitudes[keep], phi[keep], strict=True):
        data += a * np.cos(2 * np.pi * n * f0 * t + p)
    return _finish(data, fs)


def schroeder_complex(
    duration: float, fs: float, f0: float, n_harmonics: int | None = None, sign: int = 1
) -> Sound:
    """Equal-amplitude harmonic complex with Schroeder phases
    ``sign * pi * n(n+1)/N``. ``n_harmonics`` defaults to all below Nyquist."""
    n_max = int(np.ceil(fs / 2 / f0)) - 1
    N = n_max if n_harmonics is None else min(n_harmonics, n_max)
    return harmonic_complex(
        duration, fs, f0, np.arange(1, N + 1), phases="schroeder+" if sign > 0 else "schroeder-"
    )


def _bandlimited(duration, fs, f0, phase, harmonics, amplitudes, kind) -> Sound:
    harmonics = harmonics[harmonics * f0 < fs / 2]
    amplitudes = amplitudes[: len(harmonics)]
    t = time_axis(n_samples(duration, fs), fs)
    wt = 2 * np.pi * f0 * t + phase
    data = np.zeros_like(t)
    for n, a in zip(harmonics, amplitudes, strict=True):
        data += a * (np.sin(n * wt) if kind == "sin" else np.cos(n * wt))
    return _finish(data, fs)


def square_wave(duration: float, fs: float, f0: float, phase: float = 0.0, bandlimited: bool = True) -> Sound:
    """Square wave. Band-limited (odd harmonics below Nyquist) unless
    ``bandlimited=False``, which aliases badly at high f0."""
    if not bandlimited:
        t = time_axis(n_samples(duration, fs), fs)
        return _finish(np.sign(np.sin(2 * np.pi * f0 * t + phase)), fs)
    n = np.arange(1, int(fs / 2 / f0) + 1, 2)
    return _bandlimited(duration, fs, f0, phase, n, 1 / n, "sin")


def sawtooth_wave(
    duration: float, fs: float, f0: float, phase: float = 0.0, bandlimited: bool = True
) -> Sound:
    """Rising sawtooth. Band-limited unless ``bandlimited=False``."""
    if not bandlimited:
        from scipy.signal import sawtooth

        t = time_axis(n_samples(duration, fs), fs)
        return _finish(sawtooth(2 * np.pi * f0 * t + phase + np.pi), fs)
    n = np.arange(1, int(fs / 2 / f0) + 1)
    return _bandlimited(duration, fs, f0, phase, n, (-1.0) ** (n + 1) / n, "sin")


def pulse_train(duration: float, fs: float, f0: float, phase: float = 0.0, bandlimited: bool = True) -> Sound:
    """Periodic pulse (click) train.

    Band-limited: equal-amplitude cosine-phase harmonics up to Nyquist.
    Otherwise single-sample impulses at the nearest sample to each period.
    ``phase`` (radians of the fundamental) shifts the pulses in time.
    """
    if bandlimited:
        n = np.arange(1, int(np.ceil(fs / 2 / f0)))
        return _bandlimited(duration, fs, f0, phase, n, np.ones(len(n)), "cos")
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
        if hasattr(spectrum, "level_at"):  # a sigtools Spectrum
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
        Target spectrum level in dB: a :class:`~sigtools.Spectrum`, a
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
