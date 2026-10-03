"""Small numerical helpers: levels, decibels, ERB scale, time axes, RNGs."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import resample_poly

__all__ = [
    "rms",
    "amp_to_db",
    "db_to_amp",
    "freq_to_erb",
    "erb_to_freq",
    "erb_bandwidth",
    "n_samples",
    "time_axis",
    "as_rng",
    "freq_to_mel",
    "mel_to_freq",
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


def _resample_poly(data: np.ndarray, fs: float, fs_new: float, axis: int = 0, **kwargs) -> np.ndarray:
    """Polyphase resampling of ``data`` along ``axis`` from ``fs`` to ``fs_new``, the
    rate ratio approximated by a fraction with denominator at most 10000.
    ``kwargs`` go to ``scipy.signal.resample_poly`` (``padtype``)."""
    ratio = Fraction(fs_new / fs).limit_denominator(10000)
    return resample_poly(data, ratio.numerator, ratio.denominator, axis=axis, **kwargs)


def _fit_length(data: np.ndarray, length: int, mode: str = "constant") -> np.ndarray:
    """``data`` cut or padded at the end (axis 0) to ``length`` samples; ``mode``
    is ``np.pad``'s (zeros by default, ``"edge"`` repeats the last sample)."""
    if data.shape[0] >= length:
        return data[:length]
    return np.pad(data, [(0, length - data.shape[0])] + [(0, 0)] * (data.ndim - 1), mode=mode)


def _phase_ramp_delay(data: np.ndarray, shift: ArrayLike, n_fft: int, n_out: int) -> np.ndarray:
    """Delay signals along the last axis by ``shift`` samples (fractional, one per
    signal, broadcast over the leading axes) with an FFT phase ramp: band-limited
    interpolation. ``n_fft`` must leave room for the delay and the sinc tails, which
    otherwise wrap around; the first ``n_out`` samples are returned."""
    spectrum = np.fft.rfft(data, n=n_fft, axis=-1)
    freqs = np.fft.rfftfreq(n_fft)
    spectrum *= np.exp(-2j * np.pi * freqs * np.asarray(shift)[..., None])
    return np.fft.irfft(spectrum, n=n_fft, axis=-1)[..., :n_out]


def _below_nyquist(fs: float) -> float:
    """The default top band edge, 95% of the Nyquist frequency."""
    return 0.95 * fs / 2


def as_rng(rng: int | np.random.Generator | None) -> np.random.Generator:
    """Accept a seed, a Generator, or None and return a Generator."""
    return np.random.default_rng(rng)


_SLANEY_HZ_PER_MEL = 200 / 3  # linear part: 15 mel at 1000 Hz


_SLANEY_LOG_STEP = np.log(6.4) / 27  # logarithmic part: 27 mel per factor of 6.4


def freq_to_mel(freq, scale: str = "htk") -> np.ndarray:
    """Frequency [Hz] to mel.

    ``scale="htk"`` is ``2595 log10(1 + f / 700)``, the formula HTK uses
    (usually credited to O'Shaughnessy, 1987). ``"slaney"`` is the scale of
    Slaney's Auditory Toolbox and librosa: linear below 1 kHz (15 mel at
    1000 Hz) and logarithmic above it (27 mel per factor of 6.4)."""
    freq = np.asarray(freq, dtype=float)
    if scale == "htk":
        return 2595 * np.log10(1 + freq / 700)
    if scale == "slaney":
        above = 15 + np.log(np.maximum(freq, 1000) / 1000) / _SLANEY_LOG_STEP
        return np.where(freq < 1000, freq / _SLANEY_HZ_PER_MEL, above)
    raise ValueError(f"scale must be 'htk' or 'slaney', not {scale!r}")


def mel_to_freq(mel, scale: str = "htk") -> np.ndarray:
    """Mel to frequency [Hz]; the inverse of :func:`freq_to_mel`."""
    mel = np.asarray(mel, dtype=float)
    if scale == "htk":
        return 700 * (10 ** (mel / 2595) - 1)
    if scale == "slaney":
        above = 1000 * np.exp(_SLANEY_LOG_STEP * (np.maximum(mel, 15) - 15))
        return np.where(mel < 15, mel * _SLANEY_HZ_PER_MEL, above)
    raise ValueError(f"scale must be 'htk' or 'slaney', not {scale!r}")
