"""Small numerical helpers: levels, decibels, ERB scale, time axes, RNGs."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "rms",
    "amp_to_db",
    "db_to_amp",
    "freq_to_erb",
    "erb_to_freq",
    "erb_bandwidth",
    "time_axis",
    "n_samples",
    "as_rng",
]


def rms(x: ArrayLike, axis: int | None = None) -> np.ndarray | float:
    """Root-mean-square value."""
    return np.sqrt(np.mean(np.square(x), axis=axis))


def amp_to_db(x: ArrayLike, ref: float = 1.0, floor_db: float = -300.0) -> np.ndarray:
    """Amplitude (not power) to decibels: ``20*log10(|x|/ref)``, floored."""
    x = np.abs(np.asarray(x)).astype(float) / ref
    with np.errstate(divide="ignore"):
        return np.maximum(20 * np.log10(x), floor_db)


def db_to_amp(db: ArrayLike) -> np.ndarray:
    """Decibels to amplitude factor: ``10**(db/20)``."""
    return np.power(10.0, np.asarray(db, dtype=float) / 20)


def freq_to_erb(freq: ArrayLike) -> np.ndarray:
    """Frequency [Hz] to ERB-number [Cams] (Glasberg & Moore, 1990).

    The exact integral of ``1 / erb_bandwidth``, ``9.265 ln(1 + f / 228.8)``.
    The paper prints the rounded form ``21.4 log10(1 + 0.00437 f)``, which
    is about 0.3% higher (15.62 rather than 15.57 Cams at 1 kHz)."""
    return 9.265 * np.log1p(np.asarray(freq, dtype=float) / (24.7 * 9.265))


def erb_to_freq(erb: ArrayLike) -> np.ndarray:
    """ERB-number [Cams] to frequency [Hz]; inverse of :func:`freq_to_erb`."""
    return 24.7 * 9.265 * np.expm1(np.asarray(erb, dtype=float) / 9.265)


def erb_bandwidth(freq: ArrayLike) -> np.ndarray:
    """Equivalent rectangular bandwidth [Hz] of the auditory filter at ``freq``."""
    return 24.7 * (4.37 * np.asarray(freq, dtype=float) / 1000 + 1)


def n_samples(duration: float, fs: float) -> int:
    """Number of samples in ``duration`` seconds, rounded to the nearest whole
    number rather than floored (exact halves round to even, as Python's
    ``round`` does)."""
    return int(round(duration * fs))


def time_axis(n: int, fs: float) -> np.ndarray:
    """Sample times ``k/fs``. Use this instead of ``np.linspace(0, dur, n)``,
    whose spacing is ``dur/(n-1)`` rather than ``1/fs``."""
    return np.arange(n) / fs


def as_rng(rng: int | np.random.Generator | None) -> np.random.Generator:
    """Accept a seed, a Generator, or None and return a Generator."""
    return np.random.default_rng(rng)
