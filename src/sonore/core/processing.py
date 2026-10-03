"""Operations on sounds and lists of sounds."""

from __future__ import annotations

import numbers
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import butter, lfilter, sosfilt, sosfiltfilt

from sonore.core.sound import Sound
from sonore.core.utils import amp_to_db, time_axis

__all__ = [
    "match_fs",
    "match_channels",
    "match_lengths",
    "normalize",
    "concat",
    "mix",
    "relative_db",
    "butter_filter",
    "bandpass",
    "amplitude_modulate",
    "resonator",
    "antiresonator",
]

# A value that may change over time: a number, or a ``(times, values)`` pair.
Track = float | tuple[ArrayLike, ArrayLike]


def match_fs(sounds: Sequence[Sound], fs: float | None = None, mode: str = "down") -> list[Sound]:
    """Resample so all sounds share a rate: ``fs`` if given, else the lowest
    (``mode="down"``) or highest (``mode="up"``) rate in the list."""
    _check_not_empty(sounds)
    if fs is None:
        if mode not in ("down", "up"):
            raise ValueError(f"mode must be 'down' or 'up', not {mode!r}")
        rates = [s.fs for s in sounds]
        fs = {"down": min, "up": max}[mode](rates)
    return [s.resample(fs) for s in sounds]


def match_channels(sounds: Sequence[Sound]) -> list[Sound]:
    """Upmix mono sounds to the channel count of the others."""
    _check_not_empty(sounds)
    n_channels = max(s.n_channels for s in sounds)
    return [s.to_channels(n_channels) for s in sounds]


def match_lengths(sounds: Sequence[Sound], mode: str = "pad", align: str = "start") -> list[Sound]:
    """Bring sounds to one length: zero-pad to the longest (``mode="pad"``)
    or cut to the shortest (``mode="truncate"``).

    ``align`` (``start``, ``center`` or ``end``) is where each sound sits in
    the common length: padding goes after, around or before it, and cutting
    keeps its start, middle or end. With ``center``, an odd number of extra
    samples puts the odd one at the end."""
    _check_not_empty(sounds)
    if align not in ("start", "center", "end"):
        raise ValueError(f"align must be 'start', 'center' or 'end', not {align!r}")
    if mode == "pad":
        length = max(len(s) for s in sounds)
        return [s.pad_to(length, align) for s in sounds]
    if mode == "truncate":
        length = min(len(s) for s in sounds)

        def keep(sound: Sound) -> Sound:
            extra = len(sound) - length
            start = {"start": 0, "center": extra // 2, "end": extra}[align]
            return Sound(sound.data[start : start + length], sound.fs)

        return [keep(s) for s in sounds]
    raise ValueError(f"mode must be 'pad' or 'truncate', not {mode!r}")


def normalize(sounds: Sequence[Sound], rms: float | None = 1.0, peak: float | None = None) -> list[Sound]:
    """Scale each sound on its own to a target RMS (default 1) or, if
    ``peak`` is given, a target peak, as :meth:`Sound.normalize` does.

    Each sound gets its own gain, so their level differences are lost: with
    ``rms`` they all come out equally loud, with ``peak`` they all peak at
    the same value."""
    return [s.normalize(rms, peak) for s in sounds]


def _check_not_empty(sounds: Sequence[Sound]) -> None:
    if len(sounds) == 0:
        raise ValueError("need at least one sound")


def _check_fs(sounds: Sequence[Sound]) -> float:
    _check_not_empty(sounds)
    rates = {s.fs for s in sounds}
    if len(rates) > 1:
        raise ValueError(f"sampling rates differ: {sorted(rates)}; use match_fs()")
    return rates.pop()


def concat(sounds: Sequence[Sound]) -> Sound:
    """Join end to end (mono sounds are upmixed to match)."""
    fs = _check_fs(sounds)
    return Sound(np.concatenate([s.data for s in match_channels(sounds)]), fs)


def mix(sounds: Sequence[Sound], align: str = "start") -> Sound:
    """Sum sounds of possibly different lengths (zero-padded per ``align``)."""
    _check_fs(sounds)
    return sum(match_lengths(match_channels(sounds), align=align), start=0)


def relative_db(sounds: Sequence[Sound], ref: int = 0) -> list[float]:
    """Level of each sound in dB relative to ``sounds[ref]``."""
    ref_rms = sounds[ref].rms
    if ref_rms == 0:
        raise ValueError(f"the reference sounds[{ref}] is silent, so levels relative to it are undefined")
    return [float(amp_to_db(s.rms / ref_rms)) for s in sounds]


def butter_filter(
    sound: Sound,
    cutoff: float | tuple[float, float],
    btype: str = "bandpass",
    order: int = 4,
    zero_phase: bool = True,
) -> Sound:
    """Butterworth filter along time (each channel separately).

    ``btype`` is ``lowpass``, ``highpass``, ``bandpass``, or ``bandstop``.
    ``zero_phase=True`` runs it forward and backward: no phase shift, and the
    magnitude response is squared (twice the attenuation in dB, so the gain at
    the cutoff is -6 dB instead of -3 dB). This needs a sound longer than
    SciPy's padding (15 samples for a 4th-order lowpass).
    """
    sos = butter(order, cutoff, btype=btype, fs=sound.fs, output="sos")
    run_filter = sosfiltfilt if zero_phase else sosfilt
    return Sound(run_filter(sos, sound.data, axis=0), sound.fs)


def bandpass(sound: Sound, f_lo: float, f_hi: float, order: int = 4, zero_phase: bool = True) -> Sound:
    """Butterworth band-pass from ``f_lo`` to ``f_hi`` [Hz].

    Shorthand for ``butter_filter(sound, (f_lo, f_hi), "bandpass", order,
    zero_phase)``; see :func:`butter_filter`. The band edges are where the
    gain is -3 dB (one way) or -6 dB (``zero_phase=True``). ``order`` is
    SciPy's, so the band-pass has ``2 * order`` poles."""
    return butter_filter(sound, (f_lo, f_hi), "bandpass", order, zero_phase)


def amplitude_modulate(sound: Sound, f_mod: float, depth: float = 1.0, phase: float = 0.0) -> Sound:
    """Multiply by ``1 + depth*sin(2*pi*f_mod*t + phase)``.

    ``phase`` is in radians. ``depth`` 1 takes the envelope down to zero;
    above 1 the envelope changes sign (overmodulation)."""
    t = time_axis(len(sound), sound.fs)
    return sound * (1 + depth * np.sin(2 * np.pi * f_mod * t + phase))


# ------------------------------------------------------------ resonators
def _track(value: Track, t: np.ndarray, name: str = "value") -> np.ndarray | float:
    """A number as it is, or a ``(times, values)`` pair interpolated linearly
    to the times ``t`` (values held before the first and after the last)."""
    if isinstance(value, numbers.Real):
        return float(value)
    try:
        times, values = value
    except (TypeError, ValueError):
        raise TypeError(f"{name} must be a number or a (times, values) pair") from None
    times = np.asarray(times, float)
    values = np.asarray(values, float)
    if times.ndim != 1 or values.shape != times.shape or len(times) == 0:
        raise ValueError(f"{name} needs one value per time: got {times.shape} times, {values.shape} values")
    if np.any(np.diff(times) <= 0):
        raise ValueError(f"{name}'s times must increase")
    return np.interp(t, times, values)


def _resonator_coefs(f, bw, fs: float):
    """Klatt's (1980) coefficients: y[n] = A x[n] + B y[n-1] + C y[n-2]."""
    c = -np.exp(-2 * np.pi * bw / fs)
    b = 2 * np.exp(-np.pi * bw / fs) * np.cos(2 * np.pi * f / fs)
    return 1 - b - c, b, c


def _check_resonance(f, bw, fs: float) -> None:
    if np.any(np.asarray(f) < 0) or np.any(np.asarray(f) >= fs / 2):
        raise ValueError(f"resonance frequencies must be in [0, fs/2) = [0, {fs / 2:g}) Hz")
    if np.any(np.asarray(bw) <= 0):
        raise ValueError("bandwidths must be positive")


def _two_pole(x: np.ndarray, a, b, c) -> np.ndarray:
    """y[n] = a x[n] + b y[n-1] + c y[n-2] with per-sample coefficients."""
    y = np.empty_like(x)
    y1 = y2 = 0.0
    for n, (xn, an, bn, cn) in enumerate(zip(x.tolist(), a.tolist(), b.tolist(), c.tolist(), strict=True)):
        y0 = an * xn + bn * y1 + cn * y2
        y[n] = y0
        y2, y1 = y1, y0
    return y


def _two_zero(x: np.ndarray, a, b, c) -> np.ndarray:
    """y[n] = a x[n] + b x[n-1] + c x[n-2] with per-sample coefficients."""
    x1 = np.concatenate([[0.0], x[:-1]])
    x2 = np.concatenate([[0.0, 0.0], x[:-2]])[: len(x)]
    return a * x + b * x1 + c * x2


def _filter(sound: Sound, f: Track, bw: Track, inverse: bool) -> Sound:
    fs = sound.fs
    t = time_axis(len(sound), fs)
    f, bw = _track(f, t, "f"), _track(bw, t, "bw")
    _check_resonance(f, bw, fs)
    a, b, c = _resonator_coefs(f, bw, fs)
    data = sound.data
    if np.ndim(a) == 0:
        numerator, denominator = ([1 / a, -b / a, -c / a], [1.0]) if inverse else ([a], [1.0, -b, -c])
        return Sound(lfilter(numerator, denominator, data, axis=0), fs)
    a, b, c = (np.broadcast_to(coef, t.shape) for coef in (a, b, c))
    out = np.empty_like(data)
    for channel in range(data.shape[1]):
        x = data[:, channel]
        out[:, channel] = _two_zero(x, 1 / a, -b / a, -c / a) if inverse else _two_pole(x, a, b, c)
    return Sound(out, fs)


def resonator(sound: Sound, f: Track, bw: Track) -> Sound:
    """A formant: Klatt's (1980) second-order digital resonator.

    ``y[n] = A x[n] + B y[n-1] + C y[n-2]`` with ``C = -exp(-2 pi bw / fs)``,
    ``B = 2 exp(-pi bw / fs) cos(2 pi f / fs)`` and ``A = 1 - B - C``.

    This is the damped harmonic oscillator ``x'' + 2 sigma x' + w0^2 x =
    w0^2 u`` sampled: its poles are the oscillator's, mapped by ``z =
    exp(s / fs)``, with ``f`` the ringing frequency and ``bw = sigma / pi``
    the decay rate, and its gain at 0 Hz is exactly 1, as the oscillator's
    is (so ``f = 0`` gives a low-pass filter). ``f`` and ``bw`` are the peak
    and the -3 dB bandwidth only when ``bw`` is much smaller than ``f``, in
    the oscillator too. Far above the resonance the gain falls more slowly
    than the oscillator's, since a digital response repeats every ``fs``.

    ``f`` and ``bw`` are numbers, or ``(times, values)`` pairs that are
    interpolated linearly to every sample (held beyond their ends), so a
    formant can glide. The filter keeps its state while its coefficients
    change, which is what keeps a moving formant free of clicks.
    """
    return _filter(sound, f, bw, inverse=False)


def antiresonator(sound: Sound, f: Track, bw: Track) -> Sound:
    """An antiformant: the exact inverse of :func:`resonator` with the same
    ``f`` and ``bw``, a spectral zero (Klatt, 1980). Its gain at 0 Hz is 1.

    ``y[n] = (x[n] - B x[n-1] - C x[n-2]) / A``; ``f`` and ``bw`` may change
    over time as in :func:`resonator`.
    """
    return _filter(sound, f, bw, inverse=True)
